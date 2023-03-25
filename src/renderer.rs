use bytemuck::{Pod, Zeroable};
use macroquad::{
    color::*,
    texture::{draw_texture, Image, Texture2D},
    window::*,
};
use nalgebra::Vector3;
use ndarray::{Array2, Array3};
use rayon::iter::IntoParallelRefIterator;
use rayon::prelude::{IndexedParallelIterator, ParallelIterator};

use crate::world::{Intersection, World};

#[derive(Debug, Copy, Clone)]
struct Voxel {
    data: f32,
}

impl Default for Voxel {
    fn default() -> Self {
        Self {
            data: Default::default(),
        }
    }
}

#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Zeroable, Pod)]
pub struct Photon {
    pub pos: Vector3<f32>,
    pub _pad: f32,
    pub dir: Vector3<f32>,
    pub wavelength: f32,
}

impl Photon {
    fn wavelength_to_rgb(&self) -> Vector3<f32> {
        let wavelength = self.wavelength;
        let gamma = 0.8;
        let attenuation;
        let mut r = 0.0;
        let mut g = 0.0;
        let mut b = 0.0;

        if (380.0..440.0).contains(&wavelength) {
            attenuation = 0.3 + 0.7 * (wavelength - 380.0) / (440.0 - 380.0);
            r = -(wavelength - 440.0) / (440.0 - 380.0) * attenuation;
            g = 0.0;
            b = 1.0 * attenuation;
        } else if (440.0..490.0).contains(&wavelength) {
            r = 0.0;
            g = (wavelength - 440.0) / (490.0 - 440.0);
            b = 1.0;
        } else if (490.0..510.0).contains(&wavelength) {
            r = 0.0;
            g = 1.0;
            b = -(wavelength - 510.0) / (510.0 - 490.0);
        } else if (510.0..580.0).contains(&wavelength) {
            r = (wavelength - 510.0) / (580.0 - 510.0);
            g = 1.0;
            b = 0.0;
        } else if (580.0..645.0).contains(&wavelength) {
            r = 1.0;
            g = -(wavelength - 645.0) / (645.0 - 580.0);
            b = 0.0;
        } else if (645.0..=780.0).contains(&wavelength) {
            attenuation = 0.3 + 0.7 * (780.0 - wavelength) / (780.0 - 645.0);
            r = 1.0 * attenuation;
            g = 0.0;
            b = 0.0;
        }

        Vector3::new(r, g, b).map(|x| x.powf(gamma))
    }
}

const VOX_CHUNK_X: usize = 12;
const VOX_CHUNK_Y: usize = 1;
const VOX_CHUNK_Z: usize = 12;
const VOX_CHUNK_SIZE: usize = 8;
const VOX_SCALE: f32 = 0.1;
const VOX_REAL_SIZE: f32 = VOX_CHUNK_SIZE as f32 * VOX_SCALE;
const VOX_INV_SIZE: f32 = VOX_REAL_SIZE / VOX_CHUNK_SIZE as f32;

const VOX_TERR_X: f32 = VOX_CHUNK_X as f32 * VOX_REAL_SIZE;
const VOX_TERR_Y: f32 = VOX_CHUNK_Y as f32 * VOX_REAL_SIZE;
const VOX_TERR_Z: f32 = VOX_CHUNK_Z as f32 * VOX_REAL_SIZE;

pub struct Screen {
    width: usize,
    height: usize,
    texture: Texture2D,
    samp_buf: Array2<f32>,
    image: Image,
}

impl Screen {
    pub fn new() -> Self {
        let width = screen_width() as usize;
        let height = screen_height() as usize;
        let image = Image::gen_image_color(width as u16, height as u16, BLACK);
        let texture = Texture2D::from_image(&image);
        let samp_buf = Array2::<f32>::ones((width, height));

        Screen {
            width,
            height,
            image,
            texture,
            samp_buf,
        }
    }

    pub fn blend_rays<'a>(&mut self, rays: impl Iterator<Item = &'a Intersection>) {
        let mut rays_hit = 0;

        for ins in rays.filter(|ray| ray.intersection != 0) {
            let photon = ins.photon;
            let pos = ins.pos;

            let x = (pos.x * (self.width - 1) as f32).round() as usize;
            let y = (pos.y * (self.height - 1) as f32).round() as usize;

            if x >= self.width || y >= self.height {
                println!("Out of bounds: ({x}, {y})");
                continue;
            }

            let mix_fac = self.samp_buf.get_mut((x, y)).unwrap();

            let mix = 1.0 / *mix_fac;

            let x = x as u32;
            let y = y as u32;

            let mut col = self.image.get_pixel(x, y);
            let new_col = photon.wavelength_to_rgb();

            col.r = mix * new_col.x + (1.0 - mix) * col.r;
            col.g = mix * new_col.y + (1.0 - mix) * col.g;
            col.b = mix * new_col.z + (1.0 - mix) * col.b;
            col.a = 1.0;

            self.image.set_pixel(x, y, col);

            *mix_fac += 1.0;

            rays_hit += 1;
        }

        println!("Rays blended: {rays_hit}");
    }

    pub async fn present(&mut self) {
        self.texture.update(&self.image);

        draw_texture(self.texture, 0., 0., WHITE);

        next_frame().await;
    }
}

pub struct Voxels(Vec<(Voxel, Vector3<f32>)>);

pub struct State {
    pub screen: Screen,
    pub world: World,
    pub voxels: Voxels,
}

impl State {
    pub fn new() -> Self {
        let mut voxel_data = Array3::<Voxel>::default((
            VOX_CHUNK_X * VOX_CHUNK_SIZE,
            VOX_CHUNK_Y * VOX_CHUNK_SIZE,
            VOX_CHUNK_Z * VOX_CHUNK_SIZE,
        ));

        let rand = fastrand::Rng::new();
        voxel_data.indexed_iter_mut().for_each(|((x, y, z), v)| {
            v.data = 40.0 + rand.f32() * 400.0;
        });

        println!("Terrain size: {VOX_TERR_X}x{VOX_TERR_Y}x{VOX_TERR_Z}");

        let voxels = voxel_data
            .exact_chunks_mut((VOX_CHUNK_SIZE, VOX_CHUNK_SIZE, VOX_CHUNK_SIZE))
            .into_iter()
            .collect::<Vec<_>>();

        let voxel_pos_lut = voxels
            .iter()
            .enumerate()
            .map(|(i, chunk)| {
                let cx = i % VOX_CHUNK_X;
                let cy = i / VOX_CHUNK_X % VOX_CHUNK_Y;
                let cz = i / VOX_CHUNK_X / VOX_CHUNK_Y;

                let crx = cx as f32 * VOX_REAL_SIZE - VOX_TERR_X / 2.0;
                let cry = cy as f32 * VOX_REAL_SIZE - VOX_TERR_Y / 2.0;
                let crz = cz as f32 * VOX_REAL_SIZE - VOX_TERR_Z / 2.0;

                let arr = chunk
                    .indexed_iter()
                    .map(|((vx, vy, vz), _)| {
                        let vx = crx + vx as f32 * VOX_INV_SIZE;
                        let vy = cry + vy as f32 * VOX_INV_SIZE;
                        let vz = crz + vz as f32 * VOX_INV_SIZE;

                        Vector3::new(vx, vy, vz)
                    })
                    .collect::<Vec<_>>();

                Array3::from_shape_vec(chunk.dim(), arr).unwrap()
            })
            .collect::<Vec<_>>();

        let screen = Screen::new();

        let world = World::new(screen.width as f32, screen.height as f32);

        let voxels = voxels
            .par_iter()
            .zip(voxel_pos_lut.par_iter())
            .flat_map_iter(|(vox, pos)| vox.iter().cloned().zip(pos.iter().copied()))
            .collect::<Vec<_>>();

        State {
            screen,
            world,
            voxels: Voxels(voxels),
        }
    }
}

impl Voxels {
    pub fn gen_rays(&self) -> impl Iterator<Item = Intersection> + Clone + '_ {
        std::iter::once_with(|| {
            let pos = Vector3::new(
                (fastrand::f32() * 1.0 - 0.5) * VOX_TERR_X,
                (fastrand::f32() * 1.0 - 0.5) * VOX_TERR_Y,
                (fastrand::f32() * 1.0 - 0.5) * VOX_TERR_Z,
            );

            let dir = Vector3::new(
                fastrand::f32() * 2.0 - 1.0,
                fastrand::f32() * 2.0 - 1.0,
                fastrand::f32() * 2.0 - 1.0,
            );

            Intersection {
                photon: Photon {
                    pos,
                    dir: dir.normalize(),
                    wavelength: 580.0,
                    ..Default::default()
                },
                ..Default::default()
            }
        })
    }
}
