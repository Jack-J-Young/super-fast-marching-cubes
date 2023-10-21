use std::{env, num::NonZeroU32, iter, mem};
use image::{Rgb, Rgba, ImageBuffer};
use log::LevelFilter;
use env_logger::{Builder, WriteStyle};
use wgpu::{Texture, TextureView, Sampler, Extent3d, TextureFormat, TextureUsages, TextureDescriptor, TextureViewDescriptor, SamplerDescriptor, AddressMode, FilterMode, SamplerBorderColor, CompareFunction, TextureAspect, ComputePipeline, BindGroup, RenderPipeline, BindGroupLayoutDescriptor, BindGroupLayout, BindGroupLayoutEntry, ShaderStages, BindingType, StorageTextureAccess, BindGroupDescriptor, BindGroupEntry, PipelineLayoutDescriptor, PushConstantRange, ComputePipelineDescriptor, PipelineLayout, ShaderModuleDescriptor, TextureSampleType, TextureViewDimension, RenderPipelineDescriptor, VertexState, PrimitiveState, FragmentState, ColorTargetState, BlendState, ColorWrites, Features, SamplerBindingType, ImageCopyTexture, Origin3d, ImageDataLayout, BufferBinding, BufferDescriptor, BufferUsages, Buffer, BufferAddress, util::{BufferInitDescriptor, DeviceExt}};
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
    window::Window
};

fn rotate_point(p: (f32, f32), theta: f64) -> (f32, f32) {
    let xp = p.0*theta.cos() as f32 - p.1*theta.sin() as f32;
    let yp = p.0*theta.sin() as f32 + p.1*theta.cos() as f32;

    (xp, yp)
}

const CHUNK_SIZE: u32 = 32;

#[repr(C)]
#[derive(Clone)]
struct Camera {
    position: Vec<f32>,
    rotation: f32,
    scale: f32,
}

impl Camera {
    fn to_byte_arr(self) -> Vec<u8> {
        return bytemuck::cast_slice::<f32, u8>(&[self.position[0], self.position[1], self.rotation, self.scale]).to_vec();
    }
    
    fn screen_to_coord(self, pos: (f32, f32), screen_size: (f32, f32)) -> (f32, f32) {
        let centered = (pos.0 - screen_size.0/2.0, pos.1 - screen_size.1/2.0);
        let rotated = rotate_point(centered, self.rotation as f64);
        
        (rotated.0 * self.scale + self.position[0], rotated.1 * self.scale + self.position[1])
    }
}

struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,

    chunk_data: Texture,
    chunk_data_view: TextureView,

    color_buffer: Texture,
    color_buffer_view: TextureView,
    sampler: Sampler,

    ray_tracing_pipeline: ComputePipeline,
    ray_tracing_bind_group: BindGroup,
    screen_pipeline: RenderPipeline,
    screen_bind_group: BindGroup,

    camera: Camera,
    camera_buffer: Buffer,

    chunk: Chunk,

    //camera_buffer_binding: BufferBinding,

    //render_pipeline: wgpu::RenderPipeline,
    // The window must be declared after the surface so
    // it gets dropped after it as the surface contains
    // unsafe references to the window's resources.
    window: Window,
}

impl State {
    // Creating some of the wgpu types requires async code
    async fn new(window: Window) -> Self {
        let size = window.inner_size();

        // The instance is a handle to our GPU
        // Backends::all => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            dx12_shader_compiler: Default::default(),
        });
        
        // # Safety
        //
        // The surface needs to live as long as the window that created it.
        // State owns the window so this should be safe.
        let surface = unsafe { instance.create_surface(&window) }.unwrap();

        let adapter = instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            },
        ).await.unwrap();

        let adapter = instance
        .enumerate_adapters(wgpu::Backends::all())
        .find(|adapter| {
            // Check if this adapter supports our surface
            adapter.is_surface_supported(&surface)
        })
        .unwrap();
        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                features: Features::TEXTURE_BINDING_ARRAY | Features::STORAGE_RESOURCE_BINDING_ARRAY | Features::TIMESTAMP_QUERY,
                // WebGL doesn't support all of wgpu's features, so if
                // we're building for the web we'll have to disable some.
                limits: if cfg!(target_arch = "wasm32") {
                    wgpu::Limits::downlevel_webgl2_defaults()
                } else {
                    wgpu::Limits::default()
                },
                label: None,
            },
            None, // Trace path
        ).await.unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        // Shader code in this tutorial assumes an sRGB surface texture. Using a different
        // one will result all the colors coming out darker. If you want to support non
        // sRGB surfaces, you'll need to account for that when drawing to the frame.
        let surface_format = surface_caps.formats.iter()
            .copied()
            .find(|f| f.is_srgb())            
            .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        let width = size.width;
        let height = size.height;

        //println!("W: {0}, H: {1}", width, height);

        let chunk_data: Texture = device.create_texture(&TextureDescriptor {
            label: Some("rt0"),
            size: Extent3d {
                width: CHUNK_SIZE,
                height: CHUNK_SIZE,
                depth_or_array_layers: 1,
            },
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            view_formats: &[TextureFormat::Rgba8Unorm],
        });

        let chunk_data_view: TextureView = chunk_data.create_view(&wgpu::TextureViewDescriptor::default());


        let color_buffer: Texture = device.create_texture(&TextureDescriptor {
            label: Some("rt0"),
            size: Extent3d {
                width: width,
                height: height,
                depth_or_array_layers: 1,
            },
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            view_formats: &[TextureFormat::Rgba8Unorm],
        });

        let color_buffer_view: TextureView = color_buffer.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler: Sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("rt2"),
            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::Repeat,
            address_mode_w: AddressMode::Repeat,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Nearest,
            mipmap_filter: FilterMode::Nearest,
            lod_min_clamp: 0.0,
            lod_max_clamp: 0.0,
            compare: None,
            anisotropy_clamp: 1,
            border_color: None,
        });

        let camera = Camera {
            position: vec![16.0, 16.0],
            rotation: 0.0,
            scale: 1.0/20.0
        };

        let camera_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Camera Buffer"),
            usage: BufferUsages::COPY_DST | BufferUsages::STORAGE,
            contents: &camera.clone().to_byte_arr(),
        });

        let camera_buffer_binding = BufferBinding {
            buffer: &camera_buffer,
            offset: 0,
            size: None,
        };

        let chunk: Chunk = Chunk::new(CHUNK_SIZE as usize);

        //let ray_tracing_pipeline: ComputePipeline = ,
        let ray_tracing_bind_group_layout: BindGroupLayout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("rt3"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable:true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType:: StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None
                    },
                    count: None,
                }
            ]
        });

        //let binding = wgpu::BindingResource::TextureView(&color_buffer_view);

        let ray_tracing_bind_group: BindGroup = device.create_bind_group(&BindGroupDescriptor {
            label: Some("rt4"),
            layout: &ray_tracing_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&chunk_data_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&color_buffer_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(camera_buffer_binding),
                }
            ],
        });

        let ray_tracing_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("rt5"),
            bind_group_layouts: &[&ray_tracing_bind_group_layout],
            push_constant_ranges: &[],
        });

        let ray_tracing_pipeline: ComputePipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("rt6"),
            layout: Some(&ray_tracing_pipeline_layout),
            module: &device.create_shader_module(wgpu::include_wgsl!("raytracer_kernel.wgsl")),
            entry_point: "main",
        });



        let screen_bind_group_layout: BindGroupLayout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("rt32"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Sampler(SamplerBindingType::Filtering),
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false
                },
                count: None,
            }],
        });

        let screen_bind_group: BindGroup = device.create_bind_group(&BindGroupDescriptor {
            label: Some("rt42"),
            layout: &screen_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&color_buffer_view)
                }
            ],
        });

        let screen_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("rt5"),
            bind_group_layouts: &[&screen_bind_group_layout],
            push_constant_ranges: &[],
        });

        let screen_module = device.create_shader_module(wgpu::include_wgsl!("screen_shader.wgsl"));

        let screen_pipeline: RenderPipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("rt62"),
            layout: Some(&screen_pipeline_layout),
            vertex: VertexState {
                module: &screen_module,
                entry_point: "vert_main",
                buffers: &[],
            },
            fragment: Some(FragmentState {
                module: &screen_module,
                entry_point: "frag_main",
                targets: &[Some(ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList, // 1.
                strip_index_format: None,
                front_face: wgpu::FrontFace::Cw, // 2.
                cull_mode: Some(wgpu::Face::Back),
                // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            depth_stencil: None, // 1.
            multisample: wgpu::MultisampleState {
                count: 1, // 2.
                mask: !0, // 3.
                alpha_to_coverage_enabled: false, // 4.
            },
            multiview: None, // 5.
        });


    //ray_tracing_bind_group: BindGroup,
    //screen_pipeline: RenderPipeline,
    //screen_bind_group: BindGroup,

        //Shader
        // let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

        // let render_pipeline_layout =
        // device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        //     label: Some("Render Pipeline Layout"),
        //     bind_group_layouts: &[],
        //     push_constant_ranges: &[],
        // });

        // let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        //     label: Some("Render Pipeline"),
        //     layout: Some(&render_pipeline_layout),
        //     vertex: wgpu::VertexState {
        //         module: &shader,
        //         entry_point: "vs_main", // 1.
        //         buffers: &[], // 2.
        //     },
        //     fragment: Some(wgpu::FragmentState { // 3.
        //         module: &shader,
        //         entry_point: "fs_main",
        //         targets: &[Some(wgpu::ColorTargetState { // 4.
        //             format: config.format,
        //             blend: Some(wgpu::BlendState::REPLACE),
        //             write_mask: wgpu::ColorWrites::ALL,
        //         })],
        //     }),
        //     primitive: wgpu::PrimitiveState {
        //         topology: wgpu::PrimitiveTopology::TriangleList, // 1.
        //         strip_index_format: None,
        //         front_face: wgpu::FrontFace::Ccw, // 2.
        //         cull_mode: Some(wgpu::Face::Back),
        //         // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
        //         polygon_mode: wgpu::PolygonMode::Fill,
        //         // Requires Features::DEPTH_CLIP_CONTROL
        //         unclipped_depth: false,
        //         // Requires Features::CONSERVATIVE_RASTERIZATION
        //         conservative: false,
        //     },
        //     depth_stencil: None, // 1.
        //     multisample: wgpu::MultisampleState {
        //         count: 1, // 2.
        //         mask: !0, // 3.
        //         alpha_to_coverage_enabled: false, // 4.
        //     },
        //     multiview: None, // 5.
        // });

        Self {
            window,
            surface,
            device,
            queue,
            config,
            size,
            chunk_data,
            chunk_data_view,
            color_buffer,
            color_buffer_view,
            sampler,
            ray_tracing_pipeline,
            ray_tracing_bind_group,
            screen_pipeline,
            screen_bind_group,
            camera_buffer,
            camera,
            chunk,

            //render_pipeline,
        }
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::CursorMoved { device_id, position, modifiers } => {
                let screen_pos = self.camera.clone().screen_to_coord((position.x as f32, position.y as f32), (self.size.width as f32, self.size.height as f32));

                self.chunk.paint_antialiased_filled_circle(screen_pos.0, screen_pos.1, 2.5);

                true
            }

            WindowEvent::KeyboardInput {
                input: KeyboardInput {
                    state: ElementState::Pressed,
                    virtual_keycode: Some(virtual_keycode),..
                },..
            } => match virtual_keycode {
                VirtualKeyCode::Up => {
                    self.camera.position[0] += 0.5*(self.camera.rotation as f64).cos() as f32;
                    self.camera.position[1] += 0.5*(self.camera.rotation as f64).sin() as f32;
                    self.chunk.paint_antialiased_filled_circle(self.camera.position[0], self.camera.position[1], 1.5);
                    true
                },
                VirtualKeyCode::Right => {
                    self.camera.rotation += 0.05;
                    true
                },
                VirtualKeyCode::Down => {
                    self.camera.position[0] -= 0.5*(self.camera.rotation as f64).cos() as f32;
                    self.camera.position[1] -= 0.5*(self.camera.rotation as f64).sin() as f32;
                    true
                },
                VirtualKeyCode::Left => {
                    self.camera.rotation -= 0.05;
                    true
                },
                _ => false
            },
            _ => return false
        }
    }

    fn update(&mut self) {
        
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {

        //self.device.crea
        let mut command_encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Command Encoder"),
        });

        {
            let mut compute_pass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
            });

            //let mut chunk = Chunk::new(CHUNK_SIZE as usize);

            //chunk.paint_antialiased_filled_circle(64.0, 64.0, 32.0);
            //chunk.paint_antialiased_filled_circle(8.0, 8.0, 4.0);
            //chunk.paint_antialiased_filled_circle(10.0, 10.0, 16.0);

            //chunk.clone().chunk_to_image();

            let diffuse_bytes: &[u8] = &self.chunk.to_rgba8_byte_arr();

            //let diffuse_image = image::load_from_memory(diffuse_bytes).unwrap();
            //let diffuse_rgba = diffuse_image.to_rgba8();

            //use image::GenericImageView;
            //let dimensions = diffuse_image.dimensions();
            
            //command_encoder.copy_buffer_to_texture(source, destination, copy_size);

            self.queue.write_texture(ImageCopyTexture {
                texture: &self.chunk_data,
                mip_level: 0,
                origin: Origin3d { x:0, y: 0, z: 0 },
                aspect: TextureAspect::All,
            }, diffuse_bytes, ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(CHUNK_SIZE*4),
                rows_per_image: None
            },
            Extent3d { width: CHUNK_SIZE, height: CHUNK_SIZE, depth_or_array_layers: 1 });

            self.queue.write_buffer(&self.camera_buffer, 0, &self.camera.clone().to_byte_arr());

            compute_pass.set_pipeline(&self.ray_tracing_pipeline);
            compute_pass.set_bind_group(0, &self.ray_tracing_bind_group, &[]);

            //println!("w: {0}, h: {1}", self.window.inner_size().width, self.window.inner_size().height);
            //compute_pass.
            compute_pass.dispatch_workgroups(self.window.inner_size().width as u32, self.window.inner_size().height as u32, 1);
            //compute_pass.
            //compute_pass.
        }

        //let textureView: TextureView = getCurrentTexture()

        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        {
            let mut render_pass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.2,
                            a: 0.5,
                        }),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });

            //render_pass.
            render_pass.set_pipeline(&self.screen_pipeline); // 2.
            render_pass.set_bind_group(0, &self.screen_bind_group, &[]);
            //render_pass.draw(6, 1, 0, 0);
            //
            render_pass.draw(0..6, 0..1);

            //render_pass.draw_indexed(0..6, 0, 0..1);
        }

        //command_encoder.

        self.queue.submit(std::iter::once(command_encoder.finish()));

        // submit will accept anything that implements IntoIter
        //self.queue.submit([render_encoder.finish()]);
        output.present();
    
        Ok(())
    }
}


pub async fn run() {
    Builder::new()
        .filter(None, LevelFilter::Info)
        .write_style(WriteStyle::Always)
        .init();
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();

    let mut state = State::new(window).await;

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == state.window.id() => if !state.input(event) {
            match event {
                WindowEvent::CloseRequested | WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            state: ElementState::Pressed,
                            virtual_keycode: Some(VirtualKeyCode::Escape),
                            ..
                        },
                    ..
                } => *control_flow = ControlFlow::Exit,

                WindowEvent::Resized(physical_size) => {
                    state.resize(*physical_size);
                },
                WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                    // new_inner_size is &&mut so we have to dereference it twice
                    state.resize(**new_inner_size);
                },

                _ => {}
            }
        },

        Event::RedrawRequested(window_id) if window_id == state.window().id() => {
            state.update();
            match state.render() {
                Ok(_) => {}
                // Reconfigure the surface if lost
                Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                // The system is out of memory, we should probably quit
                Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                // All other errors (Outdated, Timeout) should be resolved by the next frame
                Err(e) => eprintln!("{:?}", e),
            }
        }
        Event::MainEventsCleared => {
            // RedrawRequested will only trigger once, unless we manually
            // request it.
            state.window().request_redraw();
        }

        _ => {}
    });
}


// fn chunk_to_image(chunk: &Vec<Vec<f32>>) {
//     // Create an empty image with the same dimensions as the chunk
//     let width = chunk[0].len();
//     let height = chunk.len();
//     let mut image = image::ImageBuffer::new(width as u32, height as u32);

//     // Iterate over each pixel in the chunk
//     for (y, row) in chunk.iter().enumerate() {
//         for (x, &value) in row.iter().enumerate() {
//             // Convert the value to a grayscale color
//             let color = (value * 255.0) as u8;
//             let pixel = image::Rgb([color, color, color]);

//             // Set the pixel in the image
//             image.put_pixel(x as u32, y as u32, pixel);
//         }
//     }

//     // Save the image to a file
//     image.save("chunk.png").unwrap();
// }

#[repr(C)]
#[derive(Clone)]
pub struct Chunk {
    pub data: Vec<Vec<f32>>,
    pub size: usize,
    pub updated: bool,
}

impl Chunk {
    pub fn new(size: usize) -> Self {
        let data: Vec<Vec<f32>> = iter::repeat(iter::repeat(1.0).take(size).collect()).take(size).collect();
        Self {
            data,
            size,
            updated: true,
        }
    }

    pub fn chunk_to_image(self) {
        // Create an empty image with the same dimensions as the chunk
        let width = self.data[0].len();
        let height = self.data.len();
        let mut image = image::ImageBuffer::new(width as u32, height as u32);

        // Iterate over each pixel in the chunk
        for (y, row) in self.data.iter().enumerate() {
            for (x, &value) in row.iter().enumerate() {
                // Convert the value to a grayscale color
                let color = (value * 255.0) as u8;
                let pixel = image::Rgb([color, color, color]);

                // Set the pixel in the image
                image.put_pixel(x as u32, y as u32, pixel);
            }
        }

        // Save the image to a file
        image.save("chunk.png").unwrap();
    }

    pub fn to_rgba8_byte_arr(&self) -> Vec<u8> {
        let width = self.data.len();
        let height = self.data[0].len();

        let mut img = ImageBuffer::new(width as u32, height as u32);
        
        for (x, y, pixel) in img.enumerate_pixels_mut() {
            let value = self.data[x as usize][y as usize];
            let grayscale = (value * 255.0) as u8;
            *pixel = Rgba([grayscale, grayscale, grayscale, 1.0 as u8]);
        }

        // Convert the image to RGBA8
        //let img = ;

        // Extract the byte data from the image
        let raw_data = img.into_raw();

        raw_data
    }

    pub fn get_edge(&self, dir: i32) -> Vec<Vec<f32>> {
        match dir {
            0 => vec![self.data[0].clone()],
            1 => self.data.iter().map(|x| vec![x[0]]).collect(),
            2 => vec![vec![self.data[0][0]]],
            _ => vec![],
        }
    }

    pub fn add_data(&mut self, data: &Vec<Vec<f32>>, x_offset: usize, y_offset: usize) -> &mut Self {
        self.updated = true;
        for x in 0..data.len() {
            for y in 0..data[x].len() {
                self.data[x + x_offset][y + y_offset] = data[x][y];
            }
        }

        self
    }

    pub fn paint_antialiased_filled_circle(&mut self, x: f32, y: f32, radius: f32) {
        self.updated = true;
        let radius_squared = radius * radius;
    
        for i in 0..self.data.len() {
            for j in 0..self.data[i].len() {
                let distance_squared = (i as f32 - x).powf(2.0) + (j as f32 - y).powf(2.0);
    
                if distance_squared <= radius_squared {
                    let distance = distance_squared.sqrt();
                    let alpha = 1.0 - (distance - radius).abs() / 1.0;
                    self.data[i as usize][j as usize] = alpha.max(0.0).min(self.data[i as usize][j as usize]);
                }
            }
        }
    }

    pub fn print(&self) {
        for row in self.data.iter() {
            for value in row {
                print!("{} ", value);
            }
            println!();
        }
    }
}