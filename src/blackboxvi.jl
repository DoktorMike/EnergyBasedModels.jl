using Distributions: Gaussian, gradlogpdf, logpdf, loglikelihood
using Flux.Tracker: gradient
using Flux


# Model is y = bx + a
# parameters a and b



# Arguments: log p(x|z), log p(z), log q(z;lambda)
function basic_bbvi(loglikelihood, logpdist, logqdist, nsamples)



end

