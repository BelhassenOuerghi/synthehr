"""
SynthEHR quickstart: generate, measure and decide, on a fabricated cohort.

    pip install -e .          # from the repository root
    python examples/quickstart.py

Takes a few minutes on a laptop (set QUICK = True for under a minute).
Replace `make_demo_cohort` with your own table to use it for real.
"""
from synthehr import DPTabDDPM, ReleaseGate, TabDDPM, build_evidence_pack, evaluate
from synthehr.data import patient_disjoint_split
from synthehr.demo import DISCRETE, SENSITIVE, TARGET, make_demo_cohort

QUICK = False

# 1. Data: one row per ICU stay; patients can have several stays.
cohort = make_demo_cohort(6000)
train, test = patient_disjoint_split(cohort, groups=cohort["patient_id"])
patients = train.pop("patient_id")              # an identifier, not a feature
test = test.drop(columns="patient_id")

# 2. Generate. Without privacy, a large model is the most realistic...
realistic = TabDDPM(size="large", steps=300 if QUICK else 6000,
                    timesteps=50 if QUICK else 500)
realistic.fit(train, discrete_columns=DISCRETE)

# ...under privacy, a small one is (the research found the order reverses).
private = DPTabDDPM(epsilon=10, privacy_unit="patient",
                    epochs=5 if QUICK else 200, expected_batch=1024,
                    timesteps=50 if QUICK else 500)
private.fit(train, discrete_columns=DISCRETE, groups=patients)
print(f"private model: epsilon {private.epsilon_:.2f} per patient "
      f"({private.epsilon_add_remove_:.2f} under add/remove adjacency)")

# 3. Measure each release against real rows it never saw.
gate = ReleaseGate(min_utility_ratio=0.5)
for name, gen in (("realistic", realistic), ("private", private)):
    synthetic = gen.sample(len(train))
    report = evaluate(train, test, synthetic, target=TARGET,
                      discrete_columns=DISCRETE, sensitive=SENSITIVE,
                      generator=gen, min_group_size=100)
    print(report.to_markdown())

    # 4. Decide, and write the evidence a reviewer needs.
    decision = gate.check(report)
    build_evidence_pack(report, decision, f"evidence-{name}",
                        name=f"quickstart ({name})")
    print(decision.to_markdown())
