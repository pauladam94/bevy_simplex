#import bevy_sprite::mesh2d_vertex_output::VertexOutput

struct CustomMaterial {
    x: f32,
    y: f32,
    z: f32,
    teta: f32,
    x0: f32,
    y0: f32,
    z0: f32,
    x1: f32,
    y1: f32,
    z1: f32,
    x2: f32,
    y2: f32,
    z2: f32,
};

@group(2) @binding(42) var<uniform> material: CustomMaterial;

fn sdf_circle(pos: vec3<f32>, radius: f32, center: vec3<f32>) -> f32 {
    return length(pos - center) - radius;
}

// Half Space with this equation
// a0*x + a1*y + a2*z <= a3
fn sdf_plane(p: vec3<f32>, a0: f32, a1: f32, a2: f32, a3: f32) -> f32 {
    let normal = normalize(vec3(a0, a1, a2));
    return dot(normal, p) - a3;
}

fn sdf_simplex(pos: vec3<f32>) -> f32 {
    let d1 = sdf_plane(pos, -1., 0., 0., 0.);
    let d2 = sdf_plane(pos, 0., -1., 0., 0.);
    let d3 = sdf_plane(pos, 0., 0., -1., 0.);
    let d4 = sdf_plane(pos, 1., 0., 0., 200.);
    let d5 = sdf_plane(pos, 0., 1., 0., 300.);
    let d6 = sdf_plane(pos, 1., 1., 1., 400.);
    let d7 = sdf_plane(pos, 0., 1., 3., 600.);
    return max(max(max(max(max(max(d1, d2), d3), d4), d5), d6), d7);
}

fn get_distance_from_world(pos: vec3<f32>) -> f32 {
    return sdf_simplex(pos);
    // return sdf_circle(pos, 0.3, vec3(0., 0., 0.));
}

fn calculate_normal(current_position: vec3<f32>) -> vec3<f32> {
    var SMALL_STEP = vec2<f32>(0.001, 0.0);

    var gradient_x = get_distance_from_world(current_position + SMALL_STEP.xyy) - get_distance_from_world(current_position - SMALL_STEP.xyy);
    var gradient_y = get_distance_from_world(current_position + SMALL_STEP.yxy) - get_distance_from_world(current_position - SMALL_STEP.yxy);
    var gradient_z = get_distance_from_world(current_position + SMALL_STEP.yyx) - get_distance_from_world(current_position - SMALL_STEP.yyx);

    return normalize(vec3<f32>(gradient_x, gradient_y, gradient_z));
}

fn ray_march(ray_origin: vec3<f32>, ray_direction: vec3<f32>) -> vec3<f32> {
    var total_distance_traveled = 0.0;
    var NUMBER_OF_STEPS = 128;
    var MINIMUM_HIT_DISTANCE = 0.001;
    var MAXIMUM_TRAVEL_DISTANCE = 1000.0;

    for (var i = 0; i < NUMBER_OF_STEPS; i++) {
        var current_position = ray_origin + total_distance_traveled * ray_direction;

        var distance_to_closest = get_distance_from_world(current_position);

        if distance_to_closest < MINIMUM_HIT_DISTANCE {
            var normal = calculate_normal(current_position);

            var light_position = vec3<f32>(2.0, -5.0, -3.0);

            var direction_to_light = normalize(current_position - light_position);

            var diffuse_intensity = max(0.0, dot(normal, direction_to_light));

            return vec3<f32>(1.0, 1.0, 1.0) * diffuse_intensity;
        }

        if total_distance_traveled > MAXIMUM_TRAVEL_DISTANCE {
            //No hit has occured, break out of the loop
            break;
        }

        total_distance_traveled += distance_to_closest;
    } 

    //A miss has occured so return a background color
    return vec3<f32>(0.0, 0.0, 0.0);
}

@fragment
fn fragment(mesh: VertexOutput) -> @location(0) vec4<f32> {
    // let camera_origin = camera.position;
    // let camera_pos = vec3(0., 0., 0.);
    // let sdf = sdf_circle(mesh.world_position, 2.0, vec3(1., 1., 1.));
    let posx = mesh.world_position.x / 300.;
    let posy = mesh.world_position.y / 300.;

    // let ray_origin = vec3(0., 0., 1.);
    let ray_origin = vec3(material.x, material.y, material.z);

    let ray_direction = vec3(posx, posy, -1.);

    let color = ray_march(ray_origin, ray_direction);

    return vec4(color, 1.0);
    // return vec4(posx, posy, 0.,  1.);
    // return vec4(posx, posy, 0.,  1.);
}

// float sdf(vec3 p){
//  float d1 = sdPlane(p, -1., 0., 0., 0.);
//  float d2 = sdPlane(p, 0., -1., 0., 0.);
//  float d3 = sdPlane(p, 0., 0., -1., 0.);
//  float d4 = sdPlane(p, 1., 0., 0., 200.);
//  float d5 = sdPlane(p, 0., 1., 0., 300.);
//  float d6 = sdPlane(p, 1., 1., 1., 400.);
//  float d7 = sdPlane(p, 0., 1., 3., 600.);
//  return max(max(max(max(max(max(d1,d2),d3),d4),d5),d6),d7);
// }
// 
// vec2 raymarch(vec3 ro, vec3 rd){
//  float eps = 0.0001;
//  float maxSteps = 32.;
// 
//  float t = 0.;
//  for(float i = 0.; i < maxSteps; i++){
//  float d = sdf(ro + rd * t);
//  if(d < eps) {
//  float f = i/(maxSteps-1.);
//  return vec2(t, f);
//  }
//  t += d;
//  }
//  return vec2(-1.);
// }
// 
// void mainImage(
// out vec4 fragColor,
// in vec2 fragCoord
// ) {
//  vec2 uv = fragCoord.xy/iResolution.xy;
//  uv -= 0.5;
//  uv.x *= iResolution.x/iResolution.y;
// 
//  // Define camera position (eye) and target position
//  vec3 ro = vec3(700., 1000., 600.);
//  ro.xz *= rot(iTime * 0.3);
//  vec3 targetPos = vec3(0., 0., 0.);
//     
//  // Calculate forward, up, and right vectors
//  vec3 forward = normalize(ro - targetPos);
//  vec3 up = vec3(0.0, 1.0, 0.0);
//  vec3 right = normalize(cross(forward, up));
//  up = cross(right, forward);
//  
//  vec3 rd = normalize(uv.x * right + uv.y * up - forward);
//  vec3 col = vec3(0., 0., 0.);
//  
//  vec2 hit = raymarch(ro, rd);
//  float t = hit.x;
//  float f = hit.y;
//  if(t > 0.){
//   // grayscale: darker if more iterations
//   col = vec3(1., 0., 0.) * (1. - f);
//  } else {
//   col = vec3(0., 0., 0.);
//  }
//  
//  fragColor = vec4(col,1.0);
// }
