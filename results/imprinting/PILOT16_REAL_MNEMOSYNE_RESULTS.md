# Pilot16 Mnemosyne original full-FlyWire feature-mask associative memory

Complete source-verified [actual publisher original FlyWire v783 run 38028315635](https://github.com/Azimn/Pretorius-Connectome/actions/runs/38028315635).
[Complete raw seven-stage ten-arm 450-source-event case JSON](runs/flywire-pilot16-mnemosyne-run38028315635.json).
Original case JSON SHA-256: 059b186408e56143c51b42e735c92ab416ccbb6ee37dde5e14d640a1066df10b
All TEN original source trained weight/covariance NPZ states are preserved under artifacts/imprinting/checkpoints with exact original digests in JSON.

CRITICAL: Except legacy direct Hebbian conditions, Mnemosyne learns on a **BINARY 256×256 SOURCE-FEATURE connectivity MASK extracted from the genuine full-FlyWire original 15M-edge anatomical CSR**, not on all original synapse weights. This is a feature-level biological support prior, not direct original-neuron memory modeling.
Original feature-pair structural mask permits 34310 W scalar positions, compared with **50,920 original eligible individual source/readout directed neuron edges**. Each residual model also stores 65,536 separate float64 inverse input-covariance parameters.

| Original source-linked condition | Earliest16 correct at load16 | Earliest16 correct at load317 | Newest16 correct | Familiar159 correct | Genuinely untrained fourth source cue31 correct | Native model gate correct and accepted familiar | Native absent71 wrongly accepted |
|---|---:|---:|---:|---:|---:|---:|---:|
| original_BC01_direct_hebb | 15/16 | 4/16 | 2/16 | 20/159 | 0/31 | not supported | not supported |
| original_MiniLM_top8_hebb | 10/16 | 0/16 | 0/16 | 0/159 | 0/31 | not supported | not supported |
| mnemosyne_dense_rls_original | 15/16 | 0/16 | 2/16 | 9/159 | 1/31 | 3/159 | 12/71 |
| mnemosyne_dense_rls_rewired | 16/16 | 0/16 | 3/16 | 12/159 | 0/31 | 2/159 | 12/71 |
| mnemosyne_dense_rls_random_pair_mask | 15/16 | 1/16 | 1/16 | 9/159 | 1/31 | 1/159 | 12/71 |
| mnemosyne_dense_rls_unmasked_linear | 16/16 | 2/16 | 5/16 | 26/159 | 1/31 | 3/159 | 12/71 |
| mnemosyne_sparse64_rls_original | 16/16 | 1/16 | 3/16 | 17/159 | 0/31 | 11/159 | 15/71 |
| mnemosyne_dense_delta_original | 8/16 | 0/16 | 6/16 | 7/159 | 0/31 | 0/159 | 12/71 |
| mnemosyne_dense_rls_wrong_content | 0/16 | 0/16 | 0/16 | 1/159 | 0/31 | 1/159 | 12/71 |
| mnemosyne_BC01_dense_rls_original | 15/16 | 3/16 | 7/16 | 28/159 | 1/31 | 4/159 | 9/71 |

## Explicit limits

Original MiniLM pretrained semantic cue knowledge is not encoded in fly source neurons, but borrowed externally from a frozen pretrained encoder. The learned output is only a 256D BC01 signed CONTENT vector. Actual event ID and content scores are produced only by an EXTERNAL fixed 317-source codeword diagnostic scorer. The learned model-native input coverage gate is calibrated with 62 validation absent source events, not the 71 original heldout absent events or old 159/31 probes. Its low false accept rate cannot be separated from correctly accepted known cases. Re-used source literal cue sets are exploratory, NOT independent human-written semantic confirmatory prompts. No autonomous autobiographical recollection, insect cognition or definite FlyWire topological advantage is implied.
