# RT-NRVE: Real-time Noise Reduction and Voice Enhancement using Deep Learning
## by Hanlong Chen and Zhihong Shen
## 2020/5/13

CHECK THE PRESENTATION AND DEMO HERE: https://www.youtube.com/watch?v=P_g5q2H7s7I
or presentation.mp4

RT-NRVE is a high-performance deep learning model for real-time noise reduction and voice enhancement. It can handle both stationary and non-stationary noise, and achieves superior denoising performance compared to advanced models like Wavenet.

Model Architecture
The RT-NRVE model consists of:

Input layer
Dilated convolutional network
Bidirectional GRU
Fully connected layers
Output layer
The model takes noisy spectrograms as input and outputs complex spectrogram masks that are multiplied with the noisy spectrograms to obtain the enhanced speech.

Real-time Processing
To enable real-time processing with low latency, the input audio is divided into 100ms segments. The latest segment is concatenated with the previous 4 segments to form the input to the model.

The total processing latency, including STFT, inference, and inverse STFT is around 180ms, which is below the perceptual limit of 200ms, allowing the model to operate in real-time.

Dataset
The model was trained on a dataset of 11,572 speech samples mixed with 10 types of noise at 5 different SNR levels (0dB, 5dB, 7.5dB, 10dB, 15dB). The clean speech came from the "Noisy speech database for training speech enhancement algorithms and TTS models" (NSDTSEA).

Results
RT-NRVE was evaluated on challenging speech samples with 2.5dB and 7.5dB SNR. Both spectrograms and Signal-to-Distortion Ratio (SDR) metrics show that RT-NRVE outperforms the advanced Wavenet model in denoising quality.

Acknowledgements
We thank Professor Keith Chugg and the TAs for their valuable support and feedback on this project.
