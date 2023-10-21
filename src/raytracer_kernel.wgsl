@group(0) @binding(0) var chunk_data: texture_2d<f32>;
@group(0) @binding(1) var color_buffer: texture_storage_2d<rgba8unorm,write>;
@group(0) @binding(2) var<storage> camera: Camera;

struct Camera {
    @location(0) position: vec2<f32>,
    @location(1) rotation: f32,
    @location(2) scale: f32,
}

fn bilinearInterpolationScalar(
    x: f32,
    y: f32,
    valueTopLeft: f32,
    valueTopRight: f32,
    valueBottomLeft: f32,
    valueBottomRight: f32
) -> f32 {
    // Interpolate in the x-direction first
    let valueTop = mix(valueTopLeft, valueTopRight, x);
    let valueBottom = mix(valueBottomLeft, valueBottomRight, x);
    
    // Interpolate in the y-direction
    let blendedValue = mix(valueTop, valueBottom, y);
    
    return blendedValue;
}

fn rotate_vec2(in: vec2<f32>, theta: f32) -> vec2<f32> {
    let xp = in * vec2<f32>(cos(theta), -sin(theta));
    let yp = in * vec2<f32>(sin(theta), cos(theta));

    return vec2<f32>(xp.x + xp.y, yp.x + yp.y);
}

fn cameraTransform(position: vec2<f32>, ranges: vec2<f32>) -> vec2<f32> {
    let rotated = rotate_vec2(position - ranges/2.0, camera.rotation);

    return rotated * camera.scale + camera.position;;
}

fn sample(pos: vec2<f32>) -> vec4<f32> {
    let origin = vec2<i32>(i32(pos.x), i32(pos.y));

    let a = textureLoad(chunk_data, origin                  , 0).x;
    let b = textureLoad(chunk_data, origin + vec2<i32>(1, 0), 0).x;
    let c = textureLoad(chunk_data, origin + vec2<i32>(0, 1), 0).x;
    let d = textureLoad(chunk_data, origin + vec2<i32>(1, 1), 0).x;

    let val: f32 = bilinearInterpolationScalar(pos.x - f32(origin.x), pos.y - f32(origin.y), a, b, c, d);

    var colour = vec4<f32>(1.0, 1.0, 1.0, 0.0);
    //var colour = vec4<f32>(1.0 - val, 1.0 - val, 1.0 - val, 1.0);
    if (val > 0.5) {
        colour.x = 0.0;
        colour.y = 0.0;
        colour.z = 0.0;
    }

    return colour;
}

@compute @workgroup_size(1,1,1)
fn main(@builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {

    let aliasing_count = 8;

    let screen_size: vec2<u32> = textureDimensions(color_buffer);
    let chunk_size: vec2<u32> = textureDimensions(chunk_data);

    let screen_pos: vec2<i32> = vec2<i32>(i32(GlobalInvocationID.x), i32(GlobalInvocationID.y));

    //let transx = ((f32(GlobalInvocationID.x) - f32(screen_size.x)/2.0) * cos(camera.rotation)) * camera.scale + camera.position.x;
    //let transy = ((f32(GlobalInvocationID.y) - f32(screen_size.y)/2.0) * sin(camera.rotation)) * camera.scale + camera.position.y;
    
    let transformedPosition: vec2<f32> = cameraTransform(vec2<f32>(f32(GlobalInvocationID.x), f32(GlobalInvocationID.y)), vec2<f32>(f32(screen_size.x), f32(screen_size.y)));

    let length = aliasing_count * aliasing_count;

    var blendedColor: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);

    for (var i = 0; i < aliasing_count; i++) {
        for (var j = 0; j < aliasing_count; j++) {
            
            let transformedPosition: vec2<f32> = cameraTransform(vec2<f32>(f32(GlobalInvocationID.x), f32(GlobalInvocationID.y)) + vec2<f32>(f32(i),f32(j))/f32(aliasing_count), vec2<f32>(f32(screen_size.x), f32(screen_size.y)));
            blendedColor = blendedColor + sample(transformedPosition);
        }
    }

    textureStore(color_buffer, screen_pos, blendedColor / f32(length));
}