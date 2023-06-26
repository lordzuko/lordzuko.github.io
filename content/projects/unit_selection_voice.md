---
title: "Building an English Unit Selection TTS System: Internals and a Practical Guide"
description: "Exploring unit selection TTS system"
draft: false
dateString: June 2023
tags: ["TTS", "Unit Selection", "Festival", "Speech Databases", "HTK", "sox"]
showToc: true
weight: 203
# cover:
#     image: "projects/automated-image-captioning/cover.jpg"
--- 
## 🔗 [Report](/projects/unit-selection-voice/speech_synthesis_report.pdf)


## Description

Abstract — This report explores the construction of a unit selection voice for text-to-speech (TTS) synthesis, focusing on the waveform generation stage. We develop a voice for speaking aloud dessert recipes, utilizing data collected from websites. We discuss a greedy algorithm for automatic script selection with diphone coverage maximization. We discuss the recording process, automatic corpus segmentation, and the inclusion of linguistic features and acoustic parameters in the unit selection algorithm. Two experiments with 39 non-expert listeners with 60% of native English speakers evaluated the impact of various factors on perceived naturalness and intelligibility. A comparison of the REAPER pitch marking and F0 estimation algorithm to Festival’s equivalent algorithms found no significant preference for naturalness among listeners. The second experiment assessed the effect of data quantity and domain adaptation, concluding that additional in-domain data improve naturalness and intelligibility within the domain. Overall, our findings suggest that unit selection TTS systems can produce natural and intelligible synthesized voices with minimal effort, given sufficient data with modest audio quality and phonetic coverage.

## Instructions

Record your speech and build a unit selection voice for Festival. Create variations of the voice, add domain specific data, or vary the database size. Evaluate with a listening test.
- [Instructions](https://speech.zone/exercises/build-a-unit-selection-voice/)
- [Data](https://drive.google.com/drive/folders/11qP_Hcm8PpKr8L2hLKKKfZZDOx-8PDDR?usp=sharing)
- [Qualtrics Survey](/projects/unit-selection-voice/Speech_Synthesis.qsf)

## Citations

[1] R. A. J. Clark, K. Richmond, and S. King, “Festival 2 - build your own general purpose unit selection speech synthesiser,” in Speech Synthesis Workshop, 2004.

[2] R. Clark, K. Richmond, and S. King, “Multisyn: Open-domain unit selection for the festival speech synthesis system,” Speech Communication, vol. 49, no. 4, pp. 317–330, Apr. 2007.

[3] S. Young, G. Evermann, M. Gales, T. Hain, D. Kershaw, G. Moore, J. Odell, D. Ollason, D. Povey, V. Valtchev, and P. Woodland, The HTK Book (from version 3.3), 01 2004.

[4] J. Kominek and A. W. Black, “The cmu arctic speech databases,” in Speech Synthesis Workshop, 2004.

[5] A. W. Black and K. A. Lenzo, “Optimal data selection for unit selection synthesis,” in Speech Synthesis Workshop, 2001.

[6] CSTR, The University of Edinburgh, “Speechrecorder.” [Online]. Available: https://www.cstr.ed.ac.uk/research/projects/speechrecorder/

[7] J. Taylor and K. Richmond, “Confidence Intervals for ASR-Based TTS Evaluation,” in Proc. Interspeech 2021, 2021, pp. 2791–2795.

[8] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, “Robust speech recognition via large-scale weak supervision,” 2022. [Online]. Available: https://arxiv.org/abs/2212.04356

[9] Y. Medan, E. Yair, and D. Chazan, “Super resolution pitch determination of speech signals,” IEEE Trans. Signal Process., vol. 39, pp. 40–48, 1991.

[10] D. Talkin, “Reaper: Robust epoch and pitch estimator,” https://github. com/google/REAPER, 2013.

[11] D. Jouvet and Y. Laprie, “Performance analysis of several pitch detection algorithms on simulated and real noisy speech data,” 2017 25th European Signal Processing Conference (EUSIPCO), pp. 1614–1618, 2017.

[12] F. Wilcoxon, “Individual comparisons by ranking methods,” Biometrics, vol. 1, pp. 196–202, 1945.

[13] M. Chu, C. Li, H. Peng, and E. Chang, “Domain adaptation for tts systems,” in 2002 IEEE International Conference on Acoustics, Speech, and Signal Processing, vol. 1, 2002, pp. I–453–I–456.
