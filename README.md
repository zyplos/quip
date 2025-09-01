# quip


device cpu
"a cat"
| Model | Similarity | Dimension |
|---|---|---|
| siglip-model | -0.10390240371439408 | 768 |
| siglip-different-models | -0.10389763920487129 | 768 |
| clip-models-with-projection | 0.2726726847357036 | 512 |
| clip-different-models | 0.2726726847357036 | 512 |
| jina-v1-clip | 0.2196831282026699 | 768 |
| jina-v2-auto | 0.2654551173598094 | 1024 |


device cpu
"Represent the query for retrieving evidence documents: an image of a cat"
| Model | Similarity | Dimension |
|---|---|---|
| siglip-model | -0.08097856144905545 | 768 |
| siglip-different-models | -0.08098130131410465 | 768 |
| clip-models-with-projection | 0.2729841002110684 | 512 |
| clip-different-models | 0.2729841002110684 | 512 |
| jina-v1-clip | 0.16991622004029922 | 768 |
| jina-v2-auto | 0.3084446204078343 | 1024 |


device cpu
"Represent the query for retrieving images: a cat"
| Model | Similarity | Dimension |
|---|---|---|
| siglip-model | -0.07211861867179067 | 768 |
| siglip-different-models | -0.07211913849233942 | 768 |
| clip-models-with-projection | 0.27238536999724916 | 512 |
| clip-different-models | 0.27238536999724916 | 512 |
| jina-v1-clip | 0.19278040814394864 | 768 |
| jina-v2-auto | 0.24509176141632202 | 1024 |


---


device dml
"a cat"
| Model | Similarity | Dimension |
|---|---|---|
| siglip-model | -0.1029813851315225 | 768 |
| siglip-different-models | -0.10297798197465041 | 768 |
| clip-models-with-projection | 0.27237706531958034 | 512 |
| clip-different-models | 0.27237706531958034 | 512 |
| jina-v1-clip | 0.22179266508359383 | 768 |
| jina-v2-auto | -0.03700009077474321 | 1024 |


device dml
"Represent the query for retrieving evidence documents: an image of a cat"
| Model | Similarity | Dimension |
|---|---|---|
| siglip-model | -0.07954660472104252 | 768 |
| siglip-different-models | -0.07954597589722918 | 768 |
| clip-models-with-projection | 0.2725851338612667 | 512 |
| clip-different-models | 0.2725851338612667 | 512 |
| jina-v1-clip | 0.17152283532578116 | 768 |
| jina-v2-auto | -0.0713198363425037 | 1024 |


device dml
"Represent the query for retrieving images: a cat"
| Model | Similarity | Dimension |
|---|---|---|
| siglip-model | -0.07135458878166341 | 768 |
| siglip-different-models | -0.07136756650506305 | 768 |
| clip-models-with-projection | 0.2721036938576663 | 512 |
| clip-different-models | 0.2721036938576663 | 512 |
| jina-v1-clip | 0.19523586004954271 | 768 |
| jina-v2-auto | 0.025877562103739496 | 1024 |


---


device cpu
"city"
| Model | Similarity | Dimension |
|---|---|---|
| siglip-model | -0.05744336728974016 | 768 |
| siglip-different-models | -0.05744906863898108 | 768 |
| clip-models-with-projection | 0.2651577816917307 | 512 |
| clip-different-models | 0.2651577816917307 | 512 |
| jina-v1-clip | 0.14545394944649526 | 768 |
| jina-v2-auto | 0.25125741569696625 | 1024 |


device cpu
"Represent the query for retrieving evidence documents: an image of a city"
| Model | Similarity | Dimension |
|---|---|---|
| siglip-model | -0.08330077519524116 | 768 |
| siglip-different-models | -0.08330275596122531 | 768 |
| clip-models-with-projection | 0.24913862410762422 | 512 |
| clip-different-models | 0.24913862410762422 | 512 |
| jina-v1-clip | 0.08784875971635321 | 768 |
| jina-v2-auto | 0.11918082014666966 | 1024 |

---

## scores for mismatching queries (should be low)

device cpu
"a cat"
| Model | Similarity | Dimension |
|---|---|---|
| siglip-model | -0.0769089004192731 | 768 |
| siglip-different-models | -0.07691333401627613 | 768 |
| clip-models-with-projection | 0.18373380958344712 | 512 |
| clip-different-models | 0.18373380958344712 | 512 |
| jina-v1-clip | -0.10079796871076425 | 768 |
| jina-v2-auto | 0.1744027265437938 | 1024 |

device cpu
"Represent the query for retrieving evidence documents: an image of a cat"
| Model | Similarity | Dimension |
|---|---|---|
| siglip-different-models | -0.08901958277619572 | 768 |
| siglip-model | -0.0890193875473636 | 768 |
| clip-different-models | 0.18267103072502544 | 512 |
| clip-models-with-projection | 0.18267103072502544 | 512 |
| jina-v1-clip | -0.041545165730571286 | 768 |
| jina-v2-auto | 0.14250212343530344 | 1024 |



---

To install dependencies:

```bash
bun install
```

To run:

```bash
bun run index.ts
```

This project was created using `bun init` in bun v1.2.20. [Bun](https://bun.com) is a fast all-in-one JavaScript runtime.
