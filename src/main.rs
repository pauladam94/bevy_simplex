use bevy::prelude::*;
use bevy::sprite::MaterialMesh2dBundle;
use bevy::{
    reflect::TypePath,
    render::render_resource::{AsBindGroup, ShaderRef},
    sprite::{Material2d, Material2dPlugin},
};
use bevy_egui::{EguiContexts, EguiPlugin};
use bevy_inspector_egui::quick::WorldInspectorPlugin;
use bevy_simplex::gui_simplex::egui_ui;
use bevy_simplex::gui_simplex::UiState;

fn main() {
    App::new()
        .add_plugins(
            DefaultPlugins, //     .set(AssetPlugin {
                            //     watch_for_changes_override: Some(true),
                            //     ..Default::default()
                            // })
        )
        .init_resource::<UiState>()
        .add_plugins(EguiPlugin)
        // .insert_resource(CustomMaterial::default())
        .add_plugins(WorldInspectorPlugin::new())
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
        println!("x : {}", material.1.camera);
        if keyboard_input.pressed(KeyCode::ArrowLeft) {
            material.1.camera += 0.1;
        }
        if keyboard_input.pressed(KeyCode::ArrowRight) {
            material.1.camera -= 0.1;
        }
        if material.1.camera > 1.0 {
            material.1.camera = 1.0
        } else if material.1.camera < 0. {
            material.1.camera = 0.
        }
        // if keyboard_input.pressed(KeyCode::ArrowLeft) {
        //     material.1.camera.x += 0.1;
        // }
        // if keyboard_input.pressed(KeyCode::ArrowRight) {
        //     material.1.camera.x -= 0.1;
        // }
        // if keyboard_input.pressed(KeyCode::ArrowUp) {
        //     material.1.camera.y += 0.1;
        // }
        // if keyboard_input.pressed(KeyCode::ArrowDown) {
        //     material.1.camera.y -= 0.1;
        // }
        // if keyboard_input.pressed(KeyCode::KeyA) {
        //     material.1.camera.z += 0.1;
        // }
        // if keyboard_input.pressed(KeyCode::KeyZ) {
        //     material.1.camera.z -= 0.1;
        // }
        // println!(
        //     "x : {}, y : {}, z : {}",
        //     material.1.camera.x, material.1.camera.y, material.1.camera.z
        // )
    }
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<CustomMaterial>>,
    // shader_info: Res<CustomMaterial>,
) {
    commands.spawn(Camera2dBundle::default());
    commands.spawn(MaterialMesh2dBundle {
        mesh: meshes.add(Rectangle::new(600., 600.)).into(),
        transform: Transform::default(),
        material: materials.add(CustomMaterial { camera: 2.0 }),
        ..default()
    });
}

// This is the struct that will be passed to your shader
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
struct CustomMaterial {
    #[uniform(0)]
    camera: f32,
}

impl Material2d for CustomMaterial {
    fn fragment_shader() -> ShaderRef {
        "shader.wgsl".into()
    }
}
