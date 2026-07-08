# ComfyUI RUM Anima XPred

Custom nodes for sampling RUM Anima x-pred checkpoints in ComfyUI.

## Why This Node Exists

Anima x-pred checkpoints do not output the original Flow Matching velocity directly. They output clean latent `x`, and the sampler must read out the Euler update direction as:

```text
v = (z - x_pred) / sigma
```

Regular Anima/FM ComfyUI samplers expect a velocity-prediction checkpoint, so wiring an x-pred checkpoint into those nodes uses the wrong model semantics. These nodes keep sampling in x-pred mode and use the velocity readout only inside the Euler update.

## Install

Clone or copy this repository into `ComfyUI/custom_nodes/`:

```bash
cd ComfyUI/custom_nodes
# place the ComfyUI-RUM-Anima-XPred folder here
```

Put model files under ComfyUI's normal `models/` folders:

```text
ComfyUI/models/diffusion_models/  xpred-adapter-checkpoint.safetensors
ComfyUI/models/text_encoders/     qwen_3_06b_base.safetensors
ComfyUI/models/vae/               qwen_image_vae.safetensors
ComfyUI/models/loras/             anima-lora.safetensors
```

The loader uses ComfyUI dropdowns for these files instead of absolute path text boxes.
The x-pred checkpoint is an Anima DiT checkpoint and is loaded directly through the normal Anima DiT loader, so it belongs in the same `diffusion_models` folder as ordinary Anima DiT files. A separate base DiT file is not required.

## Nodes

- `Load Anima XPred Model`
  - Loads the Anima DiT x-pred checkpoint, text encoder, and VAE.
  - Use `xpred-adapter-checkpoint.safetensors`, not `xpred-train-state.pt`.

- `Load Anima XPred LoRA`
  - Loads an Anima DiT LoRA from ComfyUI's `models/loras` folder.
  - Takes and returns `RUM_ANIMA_XPRED`, so it belongs between the model loader and sampler.

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

With a LoRA:

```text
Load Anima XPred Model -> Load Anima XPred LoRA -> Sample Anima XPred -> Preview Image / Save Image
```

Do not connect an x-pred checkpoint to a regular Anima/FM velocity sampler. The checkpoint predicts clean latent `x`, not velocity.
