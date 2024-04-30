use crate::Constraints;
use crate::LinearFunction;
use crate::Simplex;
use crate::SimplexError;
use bevy::prelude::*;
use bevy_egui::egui;
use egui::Color32;
use egui::FontFamily::Proportional;
use egui::FontId;
use egui::TextStyle::{Body, Button, Heading, Monospace, Small};

pub fn egui_ui(mut ui_state: ResMut<UiState>, ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();
    style.text_styles = [
        (Heading, FontId::new(22.0, Proportional)),
        (Body, FontId::new(18.0, Proportional)),
        (Monospace, FontId::new(14.0, Proportional)),
        (Button, FontId::new(14.0, Proportional)),
        (Small, FontId::new(10.0, Proportional)),
    ]
    .into();
    ctx.set_style(style);

    egui::SidePanel::left("Simplex Panel").show(ctx, |ui| {
        ui.vertical(|ui| {
            ui.heading("Linear Program");
            ui.horizontal(|ui| {
                egui::ComboBox::from_label("")
                    .selected_text((if ui_state.maximize { "MAX" } else { "MIN" }).to_string())
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut ui_state.maximize, true, "MAX");
                        ui.selectable_value(&mut ui_state.maximize, false, "MIN");
                    });
                ui.text_edit_singleline(&mut ui_state.function_input);
            });
            ui.text_edit_multiline(&mut ui_state.constraints_input);

            if ui.add(egui::Button::new("COMPILE")).clicked() {
                // Parse constraints
                let constraints = Constraints::compile(&mut ui_state.constraints_input).unwrap();
                // Parse linear function
                let function = ui_state
                    .function_input
                    .parse()
                    .unwrap_or(LinearFunction::zero());

                // Create simplex
                ui_state.simplex = Some(constraints.maximize(&if ui_state.maximize {
                    function
                } else {
                    -function
                }));
            }
            match &ui_state.simplex {
                Some(Ok(simplex)) => {
                    ui.heading("Values");
                    let values = simplex.current_values();
                    ui.label(
                        values
                            .iter()
                            .fold(String::new(), |acc, (v, c)| format!("{acc}{v} = {c}\n")),
                    );

                    ui.heading("State");
                    let current_state = simplex.current_state();
                    ui.colored_label(
                        Color32::RED,
                        format!("max {}", current_state.linear_function),
                    );
                    ui.label(current_state.constraints.to_string());
                }
                Some(Err(SimplexError::Unbounded)) => {
                    ui.colored_label(Color32::RED, "This program is unbounded");
                }
                None => {
                    ui.label("Press COMPILE to start the algorithm");
                }
                _ => {
                    ui.label("How did we get there ?");
                }
            }
            ui.horizontal(|ui| {
                // Previous button
                if ui.add(egui::Button::new("PREVIOUS")).clicked() {
                    if let Some(Ok(simplex)) = &mut ui_state.simplex {
                        simplex.previous_step();
                    }
                }
                // Next button
                if ui.add(egui::Button::new("NEXT")).clicked() {
                    if let Some(Ok(simplex)) = &mut ui_state.simplex {
                        match simplex.next_step(true) {
                            Ok(()) => {}
                            Err(_) => {}
                        };
                    }
                }
            })
        });
    });
}

#[derive(Resource)]
pub struct UiState {
    pub maximize: bool,

    pub function_input: String,

    pub constraints_input: String,

    pub simplex: Option<Result<Simplex, SimplexError>>,
}

impl Default for UiState {
    fn default() -> Self {
        Self {
            maximize: true,
            function_input: String::from("x + 6y + 13z"),
            constraints_input: String::from(
                "\
x <= 200\n\
y <= 300\n\
x + y + z <= 400\n\
y + 3z <= 600",
            ),
            simplex: None,
        }
    }
}
