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

