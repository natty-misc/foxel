use std::{fmt::Debug, iter::repeat_with};

use fastrand::Rng;
use macroquad::{
    color::*,
    texture::{draw_texture, Image, Texture2D},
    time,
    window::*,
};
use nalgebra::{Reflection, Vector3, Vector4, Rotation3};
use ndarray::Array2;
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

trait Geometry: Send + Sync + Debug {
    fn intersect(&self, ray: &Ray) -> Option<(f32, Vector3<f32>)>;
    fn material(&self) -> &BMat;
}

#[derive(Debug)]
struct BMat {
    albedo: Vector4<f32>,
    specular: f32,
    roughness: f32,
    emission: Vector3<f32>,
}

impl Default for BMat {
    fn default() -> Self {
        BMat {
            albedo: Vector4::new(0.0, 0.0, 0.0, 1.0),
            specular: 0.0,
            roughness: 0.25,
            emission: Vector3::new(0.0, 0.0, 0.0),
        }
    }
}

#[derive(Debug)]
struct Sphere {
    material: BMat,
    center: Vector3<f32>,
    radius: f32,
}

impl Geometry for Sphere {
    fn intersect(&self, ray: &Ray) -> Option<(f32, Vector3<f32>)> {
        let oc = ray.pos - self.center;
        let a = 1.0;
        let b = 2.0 * oc.dot(&ray.dir);
        let c = oc.dot(&oc) - self.radius * self.radius;
        let discriminant = b * b - 4.0 * a * c;
        if discriminant < 0.0 {
            return None;
        }
        let t = (-b - discriminant.sqrt()) / (2.0 * a);
        if t < 0.0 {
            return None;
        }
        Some((t, (oc + t * ray.dir).normalize()))
    }

    fn material(&self) -> &BMat {
        &self.material
    }
}

#[derive(Debug)]
struct Plane {
    material: BMat,
    normal: Vector3<f32>,
    offset: f32,
}

impl Geometry for Plane {
    fn intersect(&self, ray: &Ray) -> Option<(f32, Vector3<f32>)> {
        let denom = (-self.normal).dot(&ray.dir);
        if denom > 0.0001 {
            let t = -((-self.normal).dot(&ray.pos) + self.offset) / denom;
            if t > -0.001 {
                return Some((t, self.normal));
            }
        }

        None
    }

    fn material(&self) -> &BMat {
        &self.material
    }
}

#[derive(Debug)]
struct Ray {
    pos: Vector3<f32>,
    dir: Vector3<f32>,
}

struct World {
    geometry: Vec<Box<dyn Geometry>>,
}

fn create_world() -> World {
    let geometry: Vec<Box<dyn Geometry>> = vec![
        Box::new(Sphere {
            center: Vector3::new(0.0, 1.2, 0.0),
            radius: 2.0,
            material: BMat {
                albedo: Vector4::new(1.0, 0.0, 0.5, 1.0),
                specular: 0.9,
                ..Default::default()
            },
        }),
        Box::new(Plane {
            normal: Vector3::new(0.0, 1.0, 0.0),
            offset: -1.0,
            material: BMat {
                albedo: Vector4::new(0.5, 0.5, 0.5, 1.0),
                specular: 0.9,
                ..Default::default()
            },
        }),
        Box::new(Sphere {
            center: Vector3::new(4.0, 3.0, 4.0),
            radius: 0.1,
            material: BMat {
                emission: Vector3::new(10.0, 10.0, 10.0),
                ..Default::default()
            },
        }),
    ];

    World { geometry }
}

fn emit_ray(
    rng: &Rng,
    ray: &Ray,
    world: &World,
    mix_fac: f32,
    col: Vector4<f32>,
    iter: usize,
) -> Vector4<f32> {
    const BOUNCES: usize = 10;

    if iter >= BOUNCES {
        return col;
    }

    let hit = world
        .geometry
        .iter()
        .filter_map(|g| Some((g, g.intersect(ray)?)))
        .min_by(|(_, (a, _)), (_, (b, _))| a.partial_cmp(b).unwrap());

    if let Some((geo, (t, normal))) = hit {
        let mat = geo.material();

        let ray_n = &Ray {
            pos: ray.pos + ray.dir * t,
            dir: ray.dir - 2.0 * normal.dot(&ray.dir) * normal,
        };

        let new_col = mat.albedo * (1.0 - mat.specular)
            + mat.emission.to_homogeneous()
            + emit_ray(rng, ray_n, world, mix_fac * mat.specular, col, iter + 1) * mat.specular;

        col * (1.0 - mix_fac) + mix_fac * new_col
    } else {
        if true {
            return Vector4::new(iter as f32 / BOUNCES as f32, 0.0, 0.0, 1.0);
        }
        col * (1.0 - mix_fac) + Vector4::new(0.0, 0.0, 0.0, 1.0) * mix_fac
    }
}

#[macroquad::main("Bnuuy")]
async fn main() {
    let w = screen_width() as usize;
    let h = screen_height() as usize;

    let mut image = Image::gen_image_color(w as u16, h as u16, WHITE);

    let texture = Texture2D::from_image(&image);

    const CHUNK_SIZE: usize = 64;

    let mut color_buf = Array2::<Vector4<f32>>::default((w, h));
    let mut chunks = color_buf
        .exact_chunks_mut((CHUNK_SIZE, CHUNK_SIZE))
        .into_iter()
        .collect::<Vec<_>>();

    let mut samp_buf = Array2::<f32>::zeros((w, h));
    let mut samples = samp_buf
        .exact_chunks_mut((CHUNK_SIZE, CHUNK_SIZE))
        .into_iter()
        .collect::<Vec<_>>();

    let chunks_x = w / CHUNK_SIZE;

    let geometry = create_world();

    loop {
        let t = time::get_time() as f32;

        let rsin = (t / 4.0).sin();
        let rcos = (t / 4.0).cos();

        const N_SAMPLES: usize = 100000;

        chunks.par_iter_mut().for_each(|x| x.fill(Vector4::zeros()));

        samples.par_iter_mut().for_each(|x| {
            x.fill(1.0);
        });

        chunks
            .par_iter_mut()
            .zip(samples.par_iter_mut())
            .enumerate()
            .for_each(|(i, (c, s))| {
                let chunk_x_off = (i % chunks_x * CHUNK_SIZE) as f32;
                let chunk_y_off = (i / chunks_x * CHUNK_SIZE) as f32;

                let rng = fastrand::Rng::new();
                let xs = repeat_with(|| rng.f32()).take(N_SAMPLES);
                let ys = repeat_with(|| rng.f32()).take(N_SAMPLES);
                xs.zip(ys).for_each(|(cx, cy)| {
                    let ccx = cx * (CHUNK_SIZE as f32 - 0.0002) - 0.4999;
                    let ccy = cy * (CHUNK_SIZE as f32 - 0.0002) - 0.4999;
                    let x = chunk_x_off + ccx;
                    let y = chunk_y_off + ccy;

                    let cx = (x - (w as f32 / 2.0)) / (h as f32);
                    let cy = (1.0 - y / h as f32) - 0.5;
                    let zd = 1.2;

                    let ray_dd = Vector3::new(cx, cy, zd).normalize();
                    let rddc = ray_dd * rcos;
                    let rdds = ray_dd * rsin;

                    let pos = Vector3::new(15.0 * rsin, 0.5, 15.0 * -rcos);
                    let dir = Vector3::new(rddc.x - rdds.z, ray_dd.y, rdds.x + rddc.z).normalize();

                    let ray = &Ray { pos, dir };

                    let ix = ccx.round() as usize;
                    let iy = ccy.round() as usize;

                    let mix_fac = s.get_mut((iy, ix)).unwrap();
                    let mix = 1.0 / *mix_fac;
                    let col = c.get_mut((iy, ix)).unwrap();
                    *col =
                        mix * emit_ray(
                            &rng,
                            ray,
                            &geometry,
                            1.0,
                            Vector4::new(0.0, 0.0, 0.0, 1.0),
                            0,
                        ) + (1.0 - mix) * *col;
                    *mix_fac += 1.0;
                });
            });

        chunks.iter().enumerate().for_each(|(i, c)| {
            let chunk_x_off = i % chunks_x * CHUNK_SIZE;
            let chunk_y_off = i / chunks_x * CHUNK_SIZE;

            for (i, col) in c.iter().enumerate() {
                image.set_pixel(
                    (chunk_x_off + (i % CHUNK_SIZE)) as u32,
                    (chunk_y_off + (i / CHUNK_SIZE)) as u32,
                    col.data.0[0].into(),
                );
            }
        });

        texture.update(&image);

        draw_texture(texture, 0., 0., WHITE);

        next_frame().await
    }
}
