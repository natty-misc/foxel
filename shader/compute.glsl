#version 450

layout(local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;

struct Photon {
    vec3 pos;
    float _pad;
    vec3 dir;
    float wavelength;
};

struct Intersection {
    uint intersection;
    uint _pad;
    vec2 screen_pos;
    Photon photon;
};

layout(set = 0, binding = 0) buffer Data {
    Intersection data[];
} buf;

const float c = 3.0e8;
const float c_squared = c * c;
const float g = 6.67e-11;
const float black_hole_mass = 2.0e26;
const vec3 black_hole_pos = vec3(0.0);
const float schwarzschild_r = 2.0 * g * black_hole_mass / c_squared;
const float time_scale = 0.01;
const vec3 camera_pos = vec3(0.0, 1.5, 2.5);
const mat3 camera_rot = mat3(
    1.0000000,  0.0000000,  0.0000000,
    0.0000000, -0.8660254, -0.5000000,
    0.0000000,  0.5000000, -0.8660254
);
const float aperture = 0.01;
const float focal_length = 1.0;
const float aspect = 16.0 / 9.0;

const uint ITERATIONS = 1500;

struct ScreenIntersection {
    uint intersects;
    vec2 screen_coords;
};

ScreenIntersection test_intersection(vec3 photon_pos, vec3 photon_dir) {
    vec3 mapped = camera_rot * (photon_pos - camera_pos);

    const float EPSILON = 0.08;

    if (abs(mapped.z) > EPSILON) {
        return ScreenIntersection(0, vec2(-1.0));
    }

    if (mapped.x * mapped.x + mapped.y * mapped.y > aperture * aperture) {
        return ScreenIntersection(0, vec2(-1.0));
    }

    vec3 mapped_dir = camera_rot * photon_dir;

    const float V_EPSILON = 0.0001;

    if (mapped_dir.z > -V_EPSILON) {
        return ScreenIntersection(0, vec2(-1.0));
    }

    float x = mapped.x + mapped_dir.x / -mapped_dir.z * focal_length * 2.0;
    float y = mapped.y + mapped_dir.y / -mapped_dir.z * focal_length * 2.0;

    float sensor_width = 5.0;
    float sensor_height = sensor_width / aspect;

    if (abs(x) > sensor_width / 2.0 || abs(y) > sensor_height / 2.0) {
        return ScreenIntersection(0, vec2(-1.0));
    }

    return ScreenIntersection(1, vec2(-x / sensor_width + 0.5, -y / sensor_height + 0.5));
}

void main() {
    uint idx = gl_GlobalInvocationID.x;

    vec3 photon_pos = buf.data[idx].photon.pos;
    vec3 photon_dir = buf.data[idx].photon.dir;
    float wavelength = buf.data[idx].photon.wavelength;

    for (uint i = 0; i < ITERATIONS; i++) {
        ScreenIntersection scr_ins = test_intersection(photon_pos, photon_dir);

        if (scr_ins.intersects != 0) {
            buf.data[idx].photon.pos = photon_pos;
            buf.data[idx].photon.dir = photon_dir;
            buf.data[idx].photon.wavelength = wavelength;
            buf.data[idx].intersection = 1;
            buf.data[idx].screen_pos = scr_ins.screen_coords;
            return;
        }

        float bh_dist_old = length(black_hole_pos - photon_pos);

        photon_pos += photon_dir * time_scale;

        if (length(photon_pos - camera_pos) > float(ITERATIONS - i) * time_scale) {
            break;
        }

        vec3 bh_dist_vec = black_hole_pos - photon_pos;
        float bh_dist = length(bh_dist_vec);

        if (bh_dist < schwarzschild_r) {
            break;
        }

        vec3 bh_dir = normalize(bh_dist_vec);
        float accel = g * black_hole_mass / (bh_dist * bh_dist);
        wavelength += (accel * (bh_dist - bh_dist_old) / c_squared);
        photon_dir += bh_dir * accel / c_squared * time_scale;
        photon_dir = normalize(photon_dir);
    }

    buf.data[idx].photon.pos = photon_pos;
    buf.data[idx].photon.dir = photon_dir;
    buf.data[idx].intersection = 0;
}


