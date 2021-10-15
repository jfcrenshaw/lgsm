# Research Notebook

## Week of Oct 11, 2021

I talked with Kyle and realized I hadn't included the KLDivergence in my loss function!

- [x] Train with ColorMSE
- [ ] Add errors to simulations and fix loss functions so that errors aren't hard-coded
- [x] Implement spectral loss function
- [x] Make plots of latent variables
- [x] Implement InfoVAE (also [read the paper](https://arxiv.org/pdf/1706.02262.pdf))
- [ ] Do a hyperparameter search, including number of latent dimensions

I was thinking about why the spectral regularization doesn't seems to be working as well as I had expected.
I think part of the problem may be that I'm comparing to an interpolated version of the spectra, which may erase sharp features that have big contributions to the photometry, but are discouraged from forming during the training with specral regularization...

### Meeting with Andy, Tamas, Yashil

pre-meeting thoughts:

-  the latent space seems to be totally focused on observed properties and not on intrinsic properties. I probably need to fix this somehow... How could I do that? Maybe train a VAE restspectra -> restspectra, then train a new encoder to map (z, photometry) into that space.



## Week of Oct 4, 2021

- [x] make sure yaml merging is deeply recursive
- [x] implement flagging for config changes
- [x] create rule for plotting the SED reconstruction results
- [x] try to get jax with GPU setup
- [ ] change training so that LossFunction: False excludes that loss
- [ ] implement spectral loss function
- [ ] train model with spectral loss

## Week of Sept 27, 2021

- [x] add config values for the LGSM model to guide development (note these may change)
- [x] implement VAE
- [x] implement LGSM model
- [ ] implement loss functions  
    ^ Almost done with this - just need to implement true spectrum loss
- [ ] get default model trained  
    ^ I have trained the slope loss default model. Need to add one with spectrum loss.
- [x] Add script to plot the training history
- [ ] Try to get jax with GPU setup
- [ ] write unit test to make sure that the physics layer calculates photometry correctly
