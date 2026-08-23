# Balanced 50-query evaluation plan

The original 12-query set is intentionally preserved as `evaluation_queries.json`.
It contains 2 visual labels and 10 speech/transcript labels, so it is not suitable
for comparing modalities fairly. This plan creates 10 prompts per category.

## Vision (label from video)

1. When did I first touch the boiling pan with my towel?
2. When did I touch the noodles with my fingers?
3. When did I put the noodles into the water?
4. When did I begin chopping garlic?
5. When did I peel a garlic clove?
6. When did I add scallions?
7. When did I add the garlic to the pan?
8. When did I add sesame oil?
9. When did I add the remaining ingredients to the pan?
10. When did I shake the pan?

## Audio / transcript (timestamps already available)

1. When did I say mung bean vermicelli cooks quickly after boiling? — 0.8–7.6 s
2. When did I describe mung bean noodles as different from rice noodles? — 14.4–17.4 s
3. When did I describe the noodles as jelly-like? — 19.6–24.5 s
4. When did I say to drain the noodles? — 72.3–75.1 s
5. When did I say the noodles would go into the water? — 92.8–98.0 s
6. When did I say the noodles would stick during a rinse? — 127.4–131.3 s
7. When did I announce that I would chop garlic? — 173.8–174.9 s
8. When did I explain that garlic's heat comes from its germ? — 211.4–214.8 s
9. When did I say to add scallion for green? — 237.5–240.9 s
10. When did I say the pan had reached its smoke point? — 315.1–317.3 s

## Motion / trajectory (label from video)

1. When did I make the first large movement toward the stove?
2. When did I move the noodles toward the boiling water?
3. When did I move from handling noodles to preparing garlic?
4. When did I make a strong downward chopping motion?
5. When did I repeatedly move while peeling garlic?
6. When did I move to add scallions?
7. When did I move ingredients into the pan?
8. When did I make the first pan-shaking motion?
9. When did I make the largest pan-shaking motion?
10. When did I move to add the rest of the ingredients?

## Vision + audio (label from video and transcript)

1. When did I say mung bean vermicelli cooks quickly while showing the noodles?
2. When did I describe the jelly-like texture while handling the noodles?
3. When did I say the noodles would go in the water while putting them in?
4. When did I announce chopping garlic while starting to chop?
5. When did I explain how peeling works while peeling garlic?
6. When did I explain the garlic germ while showing it?
7. When did I say to add scallion while adding scallion?
8. When did I mention garnish while adding it?
9. When did I say to add sesame oil while pouring it?
10. When did I say the pan was at smoke point while working at the pan?

## Full multimodal (video + transcript + motion; label from video)

1. When did I explain the jelly-like noodles while moving them toward the water?
2. When did I talk about rinsing while handling the noodles?
3. When did I announce chopping garlic and begin the chopping motion?
4. When did I explain peeling garlic while manipulating the clove?
5. When did I explain the garlic germ while holding it in view?
6. When did I mention scallion while moving it into the pan?
7. When did I say to add garlic while moving it into the pan?
8. When did I describe sesame oil while pouring it into the pan?
9. When did I explain shaking cools the pan while shaking it?
10. When did I say I was ready to add the rest while moving ingredients into the pan?

## Labeling task

For every non-audio prompt, watch the video and record a relevant time range.
Use a 4–8 second range, not an exact frame. Mark uncertain labels instead of guessing.
After labels are added to `evaluation_queries.json`, rerun `scripts/run_ablation.py`.
