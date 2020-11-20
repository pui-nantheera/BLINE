This project aims to develop an objective tool <a href="https://www.youtube.com/watch?v=YzcvWdnT0sw">to determine fluid overload in children receiving dialysis</a>.
B-lines, defined as discrete laser-like vertical hyperechoic reverberation artefacts in lung ultrasounds, have been shown to correlate with extravascular lung water in symptomatic and asymptomatic adults and children on dialysis. Recent studies have shown this technique to be useful in children on dialysis.

We restored line restoration in speckle images by addressing this as a sparse estimation problem using both convex and
non-convex optimisation techniques based on the Radon transform and sparsity regularisation. This breaks into
subproblems which are solved using the alternating direction method of multipliers (ADMM), thereby achieving
line detection and deconvolution simultaneously. An automatic B-line identification method is then proposed,
using a simple local maxima technique in the Radon transform domain, associated with known clinical definitions
of line artefacts.

Figure below shows B-mode ultrasound image and detected lines (top row) and Radon transform domain representation of the restored B-mode images (bottom row). Red, yellow, blue and green lines represent the pleural lines, B-lines, A-lines and Z-lines, respectively.
<img src="https://seis.bristol.ac.uk/~eexna/research/BlineDetection/blines_results.png" width="500">
                    
Example case: B-lines shown in lung ultrasound of COVID-19 patient.
<video  loop autoplay muted  width="500">
    
    <source src="BLINE_covid.mp4"
            type="video/mp4">
            
    Please see BLINE_covid.mp4
</video>

-----------------------------------------------------------------------
This source code package includes the MATLAB source codes for B-line detection. 

Usage:

To run GUI: Blinedetection.m

To run as command line: runBlinedetection.m

Note: This code works on the images from a linear probe. The curvelinear image has to be rectified first.

-----------------------------------------------------------------------
Any usage please cite:

N. Anantrasirichai, W. Hayes, M. Allinovi, D. Bull and A. Achim, "<a href="https://research-information.bris.ac.uk/en/publications/line-detection-as-an-inverse-problem-application-to-lung-ultrasou">Line Detection as an Inverse Problem: Application to Lung Ultrasound Imaging</a>," in IEEE Transactions on Medical Imaging, vol. 36, no. 10, pp. 2045-2056, Oct. 2017.

N. Anantrasirichai, M. Allinovi, W. Hayes and A. Achim, "<a href="https://seis.bristol.ac.uk/~eexna/papers/IUS2106_Blines.pdf">Automatic B-line detection in paediatric lung ultrasound</a>," 2016 IEEE International Ultrasonics Symposium (IUS), Tours, 2016, pp. 1-4.


<a href="https://zenodo.org/badge/latestdoi/250038924"><img src="https://zenodo.org/badge/250038924.svg" alt="DOI"></a>

-----------------------------------------------------------------------
Copyright (c) Nantheera Anantrasirichai and Alin Achim
N.Anantrasirichai@bristol.ac.uk

This code is distributed under the terms of the GNU General Public License 3.0 (GNU GPLv3)

Permission to use, copy, modify, and distribute this software for any purpose without fee is hereby granted, provided that this entire notice is included in all copies of any software which is or includes a copy or modification of this software and in all copies of the supporting documentation for such software. This software is being provided "as is", without any express or implied warranty.  In particular, the authors do not make any representation or warranty of any kind concerning the merchantability of this software or its fitness for any particular purpose. 


