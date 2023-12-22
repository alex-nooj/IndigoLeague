import pathlib

from main import resume_training

resume_path = "/media/alex/USB322FD/pokemon_league/challengers/Yellow/keyboard_interrupt_564420.zip"

tag, n_steps, poke_path, model, env, team_size = resume_training(
    resume_path=pathlib.Path(resume_path),
    battle_format="gen8ou",
    rewards={
        "fainted_value": 0.0,
        "hp_value": 0.0,
        "status_value": 0.0,
        "victory_value": 1.0,
    }
)

