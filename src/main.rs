use std::sync::Arc;

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::allocator::{CommandBufferAllocator, StandardCommandBufferAllocator};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{
    DescriptorSetsCollection, PersistentDescriptorSet, WriteDescriptorSet,
};
use vulkano::device::physical::PhysicalDeviceType;
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint};
use vulkano::sync::GpuFuture;
use vulkano::VulkanLibrary;

use crate::renderer::{Photon, Screen, State};
use crate::world::Intersection;

mod renderer;
mod shader;
mod world;

#[macroquad::main("Foxel")]
async fn main() {
    let library = VulkanLibrary::new().expect("no local Vulkan library/DLL");
    let instance =
        Instance::new(library, InstanceCreateInfo::default()).expect("failed to create instance");

    let device_extensions = DeviceExtensions {
        khr_storage_buffer_storage_class: true,
        ..DeviceExtensions::empty()
    };

    let (physical, queue_family_index) = instance
        .enumerate_physical_devices()
        .unwrap()
        .filter(|p| p.supported_extensions().contains(&device_extensions))
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .position(|q| q.queue_flags.compute)
                .map(|i| (p, i as u32))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
            _ => 5,
        })
        .unwrap();

    println!("Vulkan API version: {}", physical.api_version());
    println!("Device name: {}", physical.properties().device_name);

    let (device, mut queues) = Device::new(
        physical,
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            enabled_extensions: DeviceExtensions {
                khr_storage_buffer_storage_class: true,
                ..DeviceExtensions::empty()
            },
            ..Default::default()
        },
    )
    .expect("failed to create device");

    let queue = queues.next().unwrap();

    let memory_allocator = StandardMemoryAllocator::new_default(device.clone());
    let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone());
    let command_buffer_allocator =
        StandardCommandBufferAllocator::new(device.clone(), Default::default());

    let data_initial = std::iter::repeat(Default::default())
        .take(1024 * 1024 * 4)
        .collect::<Vec<_>>();
    let data_buffer = CpuAccessibleBuffer::from_iter(
        &memory_allocator,
        BufferUsage {
            storage_buffer: true,
            ..Default::default()
        },
        false,
        data_initial,
    )
    .expect("failed to create buffer");

    let shader = shader::marcher::load(device.clone()).expect("failed to create shader module");

    let compute_pipeline = ComputePipeline::new(
        device.clone(),
        shader.entry_point("main").unwrap(),
        &(),
        None,
        |_| {},
    )
    .expect("failed to create compute pipeline");

    let layout = compute_pipeline.layout().set_layouts().get(0).unwrap();
    let set = PersistentDescriptorSet::new(
        &descriptor_set_allocator,
        layout.clone(),
        [WriteDescriptorSet::buffer(0, data_buffer.clone())],
    )
    .unwrap();

    let mut state = State::new();

    let mut rays = state.voxels.gen_rays().cycle();

    loop {
        /*
        let rays = state
            .gen_rays()
            .map(|ray| state.world.raymarch(ray))
            .filter(|ins| ins.intersection != 0)
            .collect::<Vec<_>>();
         */

        compute(
            &queue,
            &device,
            &compute_pipeline,
            set.clone(),
            &command_buffer_allocator,
            &data_buffer,
            &mut state.screen,
            &mut rays,
        );

        // state.screen.blend_rays(rays.iter());

        state.screen.present().await;
    }
}

fn compute(
    queue: &Arc<Queue>,
    device: &Arc<Device>,
    compute_pipeline: &Arc<ComputePipeline>,
    set: impl DescriptorSetsCollection + Clone,
    command_buffer_allocator: &impl CommandBufferAllocator,
    data_buffer: &Arc<CpuAccessibleBuffer<[Intersection]>>,
    screen: &mut Screen,
    data: &mut impl Iterator<Item = Intersection>,
) {
    {
        let content = &mut (*data_buffer.write().unwrap());
        for (src, dest) in data.take(1024 * 1024 * 4).zip(content.iter_mut()) {
            *dest = src;
        }
    }

    let mut builder = AutoCommandBufferBuilder::primary(
        command_buffer_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    builder
        .bind_pipeline_compute(compute_pipeline.clone())
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            compute_pipeline.layout().clone(),
            0,
            set,
        )
        .dispatch([4096, 1, 1])
        .unwrap();

    let command_buffer = builder.build().unwrap();

    let future = vulkano::sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    future.wait(None).unwrap();

    let content = data_buffer.read().unwrap();
    screen.blend_rays(content.iter());
}
