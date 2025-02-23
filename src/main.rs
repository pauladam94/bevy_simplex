use bevy::prelude::*;
use bevy::render::render_resource::AsBindGroup;
use bevy::{
    render::render_resource::ShaderRef,
    sprite::{Material2d, Material2dPlugin},
};
use bevy_egui::{EguiContexts, EguiPlugin};
// use bevy_inspector_egui::quick::WorldInspectorPlugin;
use bevy_simplex::gui_simplex::UiState;
use bevy_simplex::gui_simplex::egui_ui;

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
    mut ui_state: ResMut<UiState>,
) {
    for material in materials.iter_mut() {
        material.1.x = ui_state.camera.0;
        material.1.y = ui_state.camera.1;
        material.1.z = ui_state.camera.2;
        if keyboard_input.pressed(KeyCode::ArrowLeft) {
            ui_state.camera.0 += 1.;
        }
        if keyboard_input.pressed(KeyCode::ArrowRight) {
            ui_state.camera.0 -= 1.;
        }
        if keyboard_input.pressed(KeyCode::ArrowDown) {
            ui_state.camera.1 += 1.;
        }
        if keyboard_input.pressed(KeyCode::ArrowUp) {
            ui_state.camera.1 -= 1.;
        }
        if keyboard_input.pressed(KeyCode::KeyD) {
            ui_state.camera.2 += 1.;
        }
        if keyboard_input.pressed(KeyCode::KeyF) {
            ui_state.camera.2 -= 1.;
        }

        if let Some(Ok(simplex)) = &mut ui_state.simplex {
            if let Some(lin_prog) = simplex.first_simplex() {
                let constraints = lin_prog.constraints;
                if let Some(coeff) =
                    constraints[0].left.coefficient_var(String::from("x"))
                {
                    material.1.x0 = coeff;
                }
                if let Some(coeff) =
                    constraints[1].left.coefficient_var(String::from("x"))
                {
                    material.1.x1 = coeff;
                }
                if let Some(coeff) =
                    constraints[2].left.coefficient_var(String::from("x"))
                {
                    material.1.x2 = coeff;
                }
                if let Some(coeff) =
                    constraints[0].left.coefficient_var(String::from("y"))
                {
                    material.1.y0 = coeff;
                }
                if let Some(coeff) =
                    constraints[1].left.coefficient_var(String::from("y"))
                {
                    material.1.y1 = coeff;
                }
                if let Some(coeff) =
                    constraints[2].left.coefficient_var(String::from("y"))
                {
                    material.1.y2 = coeff;
                }
                if let Some(coeff) =
                    constraints[0].left.coefficient_var(String::from("z"))
                {
                    material.1.z0 = coeff;
                }
                if let Some(coeff) =
                    constraints[1].left.coefficient_var(String::from("z"))
                {
                    material.1.z1 = coeff;
                }
                if let Some(coeff) =
                    constraints[2].left.coefficient_var(String::from("z"))
                {
                    material.1.z2 = coeff;
                }
            }
        }
        // println!(
        //     "x : {}, y : {}, z : {}",
        //     material.1.x, material.1.y, material.1.z
        // )
    }
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<CustomMaterial>>,
    // shader_info: Res<CustomMaterial>,
) {
    // commands.spawn(MaterialMesh2dBundle {
    //     mesh: meshes.add(Rectangle::new(600., 600.)).into(),
    //     material: materials.add(CustomMaterial::default()),
    //     ..default()
    // });

    commands.spawn(Camera2d::default());
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
    teta: f32,
    #[uniform(42)]
    x0: f32,
    #[uniform(42)]
    y0: f32,
    #[uniform(42)]
    z0: f32,
    #[uniform(42)]
    x1: f32,
    #[uniform(42)]
    y1: f32,
    #[uniform(42)]
    z1: f32,
    #[uniform(42)]
    x2: f32,
    #[uniform(42)]
    y2: f32,
    #[uniform(42)]
    z2: f32,
}

impl Material2d for CustomMaterial {
    fn fragment_shader() -> ShaderRef {
        "shader.wgsl".into()
    }
}
