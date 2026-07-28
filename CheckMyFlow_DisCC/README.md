# CheckMyFlow_DisCC

Fresh implementation of CheckMyFlow / Online Distributed Conformance Checking with Footprint Matrices.

## Planned Structure

```text
checkmyflow/
  data/
    event.py
    event_log.py
    converter.py
    splitter.py

  model/
    footprint.py
    node.py
    distributed_model.py

  conformance/
    checker.py
    result.py

  experiments/
    run_checkmyflow.py
```

The implementation plan is maintained in the bachelor thesis notes:

```text
/data/.openclaw/workspace/bachelorarbeit/notes/checkmyflow-implementation-plan.md
```

## Evaluation

The experiment runner exports metrics that are intentionally close to the
`distributed-alignments` evaluation output:

```text
sum_step_loss
avg_step_loss
max_step_loss
exact_count
exact_pct
total_route
total_remote
total_queried
final_states
total_compute
elapsed_s
```

For CheckMyFlow, a mismatch counts as one step loss and a match counts as zero
loss. `final_states` is mapped to the number of learned local matrix entries.

Example:

```bash
PYTHONPATH=CheckMyFlow_DisCC python3 -m checkmyflow.experiments.run_checkmyflow \
  "datasets/SepsisCasesEventLog_1_all/Sepsis Cases - Event Log.xes/Sepsis Cases - Event Log.xes" \
  --location-key org:group \
  --training-split 0.8 \
  --random-seed 1 \
  --output-csv checkmyflow_sepsis_results.csv \
  --output-json checkmyflow_sepsis_results.json
```

```cmd
set PYTHONPATH=CheckMyFlow_DisCC

python -m checkmyflow.experiments.run_checkmyflow "datasets\SepsisCasesEventLog_1_all\Sepsis Cases - Event Log.xes\Sepsis Cases - Event Log.xes" --location-key org:group --training-split 0.8 --random-seed 1 --output-csv checkmyflow_sepsis_results.csv --output-json checkmyflow_sepsis_results.json
```

