SD FaceFusion, by RunDiffusion
==========

![image](https://github.com/runnitai/sd_facefusion/assets/1633844/bbfa6b69-c8db-4089-81df-048dd6fe89a5)

> Next generation face swapper and enhancer extension for Automatic1111, based on the original FaceFusion project.

[![Build Status](https://img.shields.io/github/actions/workflow/status/facefusion/facefusion/ci.yml.svg?branch=master)](https://github.com/facefusion/facefusion/actions?query=workflow:ci)
![License](https://img.shields.io/badge/license-MIT-green)


Preview
-------

<img width="1199" alt="image" src="https://github.com/runnitai/sd_facefusion/assets/1633844/7534bc81-1305-427e-b6e8-1b6e0617397c">
[For the most optimal cloud experience for Automatic, SD Facefusion, and everything else, check out Rundiffusion](https://rundiffusion.com/)


Features
--------
Job Queue to allow pre-staging multiple jobs and letting it run.

Multiple reference face selection for better face consistency.

Optional target video loading from URL or direct path. Works for Youtube (see below) and other video files where the type is directly detectable from the URL.

Automatic threading for optimal job performance regardless of GPU.

Auto-downloading of models, and integration with Automatic1111 for outputs and model storage.

Live job updates with preview images while processing.



Installation
------------

Install from the URL for this repo. All models will be downloaded on first run.

For youtube downloading, you need to manually patch the pytube libarary in the venv to fix some age restriced error.
https://github.com/pytube/pytube/pull/1790/files

Disclaimer
----------

We acknowledge the unethical potential of FaceFusion and are resolutely dedicated to establishing safeguards against such misuse. This program has been engineered to abstain from processing inappropriate content such as nudity, graphic content and sensitive material.

It is important to note that we maintain a strong stance against any type of pornographic nature and do not collaborate with any websites promoting the unauthorized use of our software.

Users who seek to engage in such activities will face consequences, including being banned from our community. We reserve the right to report developers on GitHub who distribute unlocked forks of our software at any time.


Documentation
-------------

Read the [documentation](https://docs.facefusion.io) for a deep dive.
