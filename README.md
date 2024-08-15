# DreamerV3

https://arxiv.org/abs/2301.04104

Want to do well on Atari100k (`pip install gym[atari] autorom[accept-rom-license]`), though BSuite (`pip install bsuite`) looks interesting too.

This is designed to run on a tinybox, either red or green, with just `./train.py`

## Process

1. Run https://github.com/danijar/dreamerv3 to train a model that plays Pong
2. Get that model loaded into tinygrad and running, both the policy model and decoder
3. Get fine tuning working
4. Get full training working

## delta-iris

Might be a better choice, the repo is a lot easier to read. https://github.com/vmicheli/delta-iris

Three models:
* actor_critic (two copies, model and target_model)
* world_model
  * transformer takes in (frames_emb x1, act_tokens_emb x1, latents_emb x4) x many
  * frame_cnn (FrameEncoder), output 4 channels
* tokenizer
  * frame_cnn (FrameEncoder), output 16 channels
  * encoder is 7 channels, 3 for prev_frame, 1 for action, and 3 for frame (FrameEncoder), output 64 channels for quantizer
  * decoder is 84 channels, 16 for prev_frame, 4 for action, and 64 for latents. it outputs an image (FrameDecoder)
  * quantizer

Training:
* Happens in three distinct phases
  * First, tokenizer is trained. It outputs 4 (from a vocab of 1024, codebook dim of 64) tokens per delta image
    * q = encoder(img_0, encoder_act_emb(a), img_1)
    * decoder(frame_cnn(img_0), decoder_act_emb(a), q)
  * Then, world model is trained
    * transformer([frame_cnn(img_0), act_emb(a), latents_emb(tokens_from_encoder), ...])
  * Last, actor critic is trained (in world model)

