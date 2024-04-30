use bevy::prelude::*;
use bevy::sprite::MaterialMesh2dBundle;
use bevy::{
    reflect::TypePath,
    render::render_resource::{AsBindGroup, ShaderRef},
    sprite::{Material2d, Material2dPlugin},
};
use bevy_egui::{EguiContexts, EguiPlugin};
// use bevy_inspector_egui::quick::WorldInspectorPlugin;
use bevy_simplex::gui_simplex::egui_ui;
use bevy_simplex::gui_simplex::UiState;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .init_resource::<UiState>()
        .add_plugins(EguiPlugin)
        // .insert_resource(CustomMaterial::default())
        // .add_plugins(WorldInspectorPlugin::new())
        // .add_plugins(AssetInspectorPlugin::<CustomMaterial>::default())
        // .add_plugins(DefaultInspectorConfigPlugin::name)
        .add_plugins(Material2dPlugin::<CustomMaterial>::default())
        .add_systems(Update, ui_example_system)
        .add_systems(Startup, setup)
        .add_systems(Update, move_camera)
        .run();
}

fn ui_example_system(ui_state: ResMut<UiState>, mut contexts: EguiContexts) {
    egui_ui(ui_state, contexts.ctx_mut())
}

fn move_camera(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut materials: ResMut<Assets<CustomMaterial>>,
) {
    for material in materials.iter_mut() {
        // println!("x : {}", material.1.camera);
        // if keyboard_input.pressed(KeyCode::ArrowLeft) {
        //     material.1.camera += 0.1;
        // }
        // if keyboard_input.pressed(KeyCode::ArrowRight) {
        //     material.1.camera -= 0.1;
        // }
        // if material.1.camera > 1.0 {
        //     material.1.camera = 1.0
        // } else if material.1.camera < 0. {
        //     material.1.camera = 0.
        // }
        if keyboard_input.pressed(KeyCode::ArrowLeft) {
            material.1.x += 0.1;
        }
        if keyboard_input.pressed(KeyCode::ArrowRight) {
            material.1.x -= 0.1;
        }
        if keyboard_input.pressed(KeyCode::ArrowUp) {
            material.1.y += 0.1;
        }
        if keyboard_input.pressed(KeyCode::ArrowDown) {
            material.1.y -= 0.1;
        }
        if keyboard_input.pressed(KeyCode::KeyA) {
            material.1.z += 0.1;
        }
        if keyboard_input.pressed(KeyCode::KeyZ) {
            material.1.z -= 0.1;
        }
        println!(
            "x : {}, y : {}, z : {}",
            material.1.x, material.1.y, material.1.z
        )
    }
}

#[derive(Component)]
struct Movable;

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<CustomMaterial>>,
    // shader_info: Res<CustomMaterial>,
) {
    commands
        .spawn(MaterialMesh2dBundle {
            mesh: meshes.add(Rectangle::new(600., 600.)).into(),
            material: materials.add(CustomMaterial::default()),
            ..default()
        })
        .insert(Movable);

    commands.spawn(Camera2dBundle::default());
}

// This is the struct that will be passed to your shader
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone, Default)]
struct CustomMaterial {
    #[uniform(42)]
    x: f32,
    #[uniform(42)]
    y: f32,
    #[uniform(42)]
    z: f32,
    #[uniform(42)]
    x1: f32,
}

impl Material2d for CustomMaterial {
    fn fragment_shader() -> ShaderRef {
        "shader.wgsl".into()
    }
}

// Materials are used alongside [`Material2dPlugin`] and [`MaterialMesh2dBundle`]
// to spawn entities that are rendered with a specific [`Material2d`] type. They serve as an easy to use high level
// way to render [`Mesh2dHandle`] entities with custom shader logic.
//
// Material2ds must implement [`AsBindGroup`] to define how data will be transferred to the GPU and bound in shaders.
// [`AsBindGroup`] can be derived, which makes generating bindings straightforward. See the [`AsBindGroup`] docs for details.
//
// # Example
//
// Here is a simple Material2d implementation. The [`AsBindGroup`] derive has many features. To see what else is available,
// check out the [`AsBindGroup`] documentation.
// ```
// # use bevy_sprite::{Material2d, MaterialMesh2dBundle};
// # use bevy_ecs::prelude::*;
// # use bevy_reflect::TypePath;
// # use bevy_render::{render_resource::{AsBindGroup, ShaderRef}, texture::Image, color::Color};
// # use bevy_asset::{Handle, AssetServer, Assets, Asset};
//
// #[derive(AsBindGroup, Debug, Clone, Asset, TypePath)]
// pub struct CustomMaterial {
//     // Uniform bindings must implement `ShaderType`, which will be used to convert the value to
//     // its shader-compatible equivalent. Most core math types already implement `ShaderType`.
//     #[uniform(0)]
//     color: Color,
//     // Images can be bound as textures in shaders. If the Image's sampler is also needed, just
//     // add the sampler attribute with a different binding index.
//     #[texture(1)]
//     #[sampler(2)]
//     color_texture: Handle<Image>,
// }
//
// // All functions on `Material2d` have default impls. You only need to implement the
// // functions that are relevant for your material.
// impl Material2d for CustomMaterial {
//     fn fragment_shader() -> ShaderRef {
//         "shaders/custom_material.wgsl".into()
//     }
// }
//
// // Spawn an entity using `CustomMaterial`.
// fn setup(mut commands: Commands, mut materials: ResMut<Assets<CustomMaterial>>, asset_server: Res<AssetServer>) {
//     commands.spawn(MaterialMesh2dBundle {
//         material: materials.add(CustomMaterial {
//             color: Color::RED,
//             color_texture: asset_server.load("some_image.png"),
//         }),
//         ..Default::default()
//     });
// }
// ```
// In WGSL shaders, the material's binding would look like this:
//
// ```wgsl
// struct CustomMaterial {
//     color: vec4<f32>,
// }
//
// @group(2) @binding(0) var<uniform> material: CustomMaterial;
// @group(2) @binding(1) var color_texture: texture_2d<f32>;
// @group(2) @binding(2) var color_sampler: sampler;
// ```
