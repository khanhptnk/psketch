# psketch


### Quick start

1. Clone repo: `git clone https://github.com/khanhptnk/psketch.git`

2. Run experiment with primitive language teacher (first-attempt algo): 

`python train.py -config configs/experiments/primitive_language.yaml`

3. Training progress will be logged to `experiments/primitive_language/run.log`

You must delete `experiments/primitive_language` to able to run the experiment again, or specify a different experiment directory using the `-name` flag (see `flags.py`). 

### Visualization


### Code structure

`trainers`: training algorithms
   - `imitation.py`: BC/DAgger algorithm (see `policy_mix` in `configs/experiments/imitation.yaml` config file, set `init_rate` = 1 for BC, = 0 for DAgger)
   - `primitive_language.py`: story 1 algo
   - `interactive_primitive_language`: see student
   - `active_primitive_language`: see student

`students`: students
   - `imitation.py`: BC/DAgger student
   - `primitive_language`: story 1 student (no interaction during task execution)
   - `interactive_primitive_language`: student that queries teacher in every time step (DAgger version of primtive language student)
   - `active_primitive_language`: interactive student that queries only when uncertain

`models`: student models (e.g. LSTM encoder-decoder)

`teachers`: teachers
   - `demonstration.py`: provides action demonstration
   - `primitive_language.py`: instruct and describe using only primitive language
   - `active_primitive_language.py`: similar to `primitive_language` but only gives next-action instructions


