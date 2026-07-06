# ComfyUI RUM Anima XPred

Custom nodes for sampling RUM Anima x-pred checkpoints in ComfyUI.

Repository: https://github.com/leafmoone/ComfyUI-RUM-Anima-XPred

## Why This Node Exists

Anima x-pred checkpoints do not output the original Flow Matching velocity directly. They output clean latent `x`, and the sampler must read out the Euler update direction as:

```text
v = (z - x_pred) / sigma
```

Regular Anima/FM ComfyUI samplers expect a velocity-prediction checkpoint, so wiring an x-pred checkpoint into those nodes uses the wrong model semantics. These nodes keep sampling in x-pred mode and use the velocity readout only inside the Euler update.

## Install

Clone this repository into `ComfyUI/custom_nodes/`:

```bash
git clone https://github.com/leafmoone/ComfyUI-RUM-Anima-XPred.git
```

The nodes need the RUM Anima X-Pred backend repository because model loading and sampling reuse `src/rum_xpred`. If the backend is not at `/root/shared-nvme/RUM-anima-xpred` or a sibling directory, set:

```bash
export RUM_ANIMA_XPRED_ROOT=/path/to/RUM-anima-xpred
```

Put model files under ComfyUI's normal `models/` folders:

```text
ComfyUI/models/anima_xpred/       xpred-adapter-checkpoint.safetensors
ComfyUI/models/text_encoders/     qwen_3_06b_base.safetensors
ComfyUI/models/vae/               qwen_image_vae.safetensors
```

The loader uses ComfyUI dropdowns for these files instead of absolute path text boxes.
The x-pred checkpoint is loaded directly through the normal Anima DiT loader, so a separate base DiT file is not required.

## Nodes

- `Load Anima XPred Model`
  - Loads the Anima DiT x-pred checkpoint, text encoder, and VAE.
  - Use `xpred-adapter-checkpoint.safetensors`, not `xpred-train-state.pt`.

- `Sample Anima XPred`
  - Runs the dedicated x-pred sampler with `heun` or `euler`:
    `v = (z - x_pred) / sigma`
  - The Heun mode follows JLT's predictor-corrector structure and uses Euler for the final step into `sigma=0`.
  - Decodes the final latent with the Anima/Qwen VAE.
  - Outputs a ComfyUI `IMAGE` plus the raw latent dictionary.

## Minimal Workflow

Connect:

```text
Load Anima XPred Model -> Sample Anima XPred -> Preview Image / Save Image
```

Do not connect an x-pred checkpoint to a regular Anima/FM velocity sampler. The checkpoint predicts clean latent `x`, not velocity.
