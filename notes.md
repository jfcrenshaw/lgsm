# Research Notebook

## Week of Oct 4

- [x] make sure yaml merging is deeply recursive
- [x] implement flagging for config changes
- [x] create rule for plotting the SED reconstruction results
- [x] try to get jax with GPU setup
- [ ] change training so that LossFunction: False excludes that loss
- [ ] implement spectral loss function
- [ ] train model with spectral loss

## Week of Sept 27

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
