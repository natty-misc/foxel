use bytemuck::{Pod, Zeroable};
use nalgebra::{Unit, UnitQuaternion, Vector2, Vector3};

use crate::renderer::Photon;

struct Camera {
    pos: Vector3<f32>,
    dir: Unit<Vector3<f32>>,
    up: Unit<Vector3<f32>>,
    aspect: f32,
    aperture: f32,
    focal_length: f32,
    rot: UnitQuaternion<f32>,
}

impl Camera {
    fn new(
        pos: Vector3<f32>,
        dir: Unit<Vector3<f32>>,
        up: Unit<Vector3<f32>>,
        aspect: f32,
        aperture: f32,
        focal_length: f32,
    ) -> Self {
        let rot = UnitQuaternion::rotation_between(&dir, &Vector3::z()).unwrap();

        Self {
            pos,
            dir,
            up,
            aspect,
            aperture,
            focal_length,
            rot,
        }
    }

    fn collision(&self, photon: &Photon) -> Option<Vector2<f32>> {
        let mapped = self.rot * (photon.pos - self.pos);

        const EPSILON: f32 = 0.08;

        if mapped.z.abs() > EPSILON {
            return None;
        }

        if mapped.x * mapped.x + mapped.y * mapped.y > self.aperture * self.aperture {
            return None;
        }

        let mapped_dir = self.rot * photon.dir;

        const V_EPSILON: f32 = 0.0001;

        if mapped_dir.z > -V_EPSILON {
            return None;
        }

        let x = mapped.x + mapped_dir.x / -mapped_dir.z * self.focal_length * 2.0;
        let y = mapped.y + mapped_dir.y / -mapped_dir.z * self.focal_length * 2.0;

        let sensor_width = 5.0;
        let sensor_height = sensor_width / self.aspect;

        if x.abs() > sensor_width / 2.0 || y.abs() > sensor_height / 2.0 {
            return None;
        }

        Some(Vector2::new(
            -x / sensor_width + 0.5,
            -y / sensor_height + 0.5,
        ))
    }
}

pub struct World {
    camera: Camera,
    black_hole_pos: Vector3<f32>,
    black_hole_mass: f32,
}

#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Zeroable, Pod)]
pub struct Intersection {
    pub intersection: u32,
    pub _pad: u32,
    pub pos: Vector2<f32>,
    pub photon: Photon,
}

impl World {
    pub fn new(screen_width: f32, screen_height: f32) -> World {
        let world = World {
            camera: Camera::new(
                Vector3::new(0.0, 1.5, 2.5),
                Unit::new_normalize(Vector3::new(0.0, -1.5, -2.5)),
                Unit::new_normalize(Vector3::y()),
                screen_width / screen_height,
                0.02,
                1.0,
            ),
            black_hole_pos: Vector3::new(0.0, 0.0, 0.0),
            black_hole_mass: 2.0e26,
        };

        world
    }

    const fn c(&self) -> f32 {
        3e8
    }

    fn c_squared(&self) -> f32 {
        self.c() * self.c()
    }

    fn g(&self) -> f32 {
        6.67e-11
    }

    fn schwarzschild_radius(&self) -> f32 {
        2.0 * self.g() * self.black_hole_mass / (self.c() * self.c())
    }

    pub fn raymarch(&self, mut photon: Photon) -> Intersection {
        const ITERATIONS: usize = 600;

        let scharzschild_r = self.schwarzschild_radius();
        let time_scale = 0.015;

        for i in 0..ITERATIONS {
            if let Some(pos) = self.camera.collision(&photon) {
                return Intersection {
                    intersection: true.into(),
                    pos,
                    photon,
                    ..Default::default()
                };
            }

            photon.pos += photon.dir * time_scale;

            if photon.pos.metric_distance(&self.camera.pos) > (ITERATIONS - i) as f32 * time_scale {
                break;
            }

            let bh_dist = photon.pos.metric_distance(&self.black_hole_pos);
            if bh_dist < scharzschild_r {
                break;
            }

            let bh_dir = (self.black_hole_pos - photon.pos).normalize();
            let bh_dist2 = bh_dist * bh_dist;
            let a = self.g() * self.black_hole_mass / bh_dist2;
            photon.dir += bh_dir * a / self.c_squared() * time_scale;
            photon.dir.normalize_mut();
        }

        Intersection {
            intersection: false.into(),
            photon,
            ..Default::default()
        }
    }
}
