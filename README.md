# Enhanced Low SNR Radio Signal Classification using Deep Learning
 Final Project for AI Wireless 2020 in National Chiao Tung University
 *Top final project in the course*
 
## Contributors
[Farhan Tandia](https://github.com/farhantandia)<br>
[Ivan Surya Hutomo](https://github.com/alexivaner)<br>
 
# Abstract
The ability to classify signals is an important task that holds the opportunity for many different applications. Previously to classify the signal, we should decompose the signal using FT (Fourier Transform), SIFT, MFCC, or another handcrafting method using statistical modulation features. In the past five years, we have seen rapid disruption occurring based on the improved neural network architectures, algorithms, and optimization techniques collectively known as deep learning (DL). It turns out that state of the art deep learning methods can be applied to the same problem of signal classification and shows excellent results while completely avoiding the need for difficult handcrafted feature selection. In 2018, people use ResNet as a state of the art of computer vision to classify radio communication signals. But ResNet only still fail to distinguish signal with low SNR condition. They only work well on a signal with high SNR Conditions. After two years, deep learning already improved a lot and many methods have become the new state of the art that we could apply for radio signal classification. Hence, we propose a new state of the art method to better classifying radio-signal network that both works on a signal with low noise (High SNR) and signal with high noise (Low SNR). Our works even will work using only RAW signal without the need preprocessing or denoising the noisy signal.


# Goals
<img src="https://github.com/alexivaner/Deep-Learning-Based-Radio-Signal-Classification/raw/main/Submission/Final/Kinds%20of%20Signal.png" width="500"><br>

* Classify 24 kinds of signal and get higher accuracy in lower SNR value.
* Design a new deep learning architecture and try to get the comparable results in terms of accuracy with state of the art or even better.
* Create End-to-end Deep Learning Model System (using only RAW signal).

* 24 Kinds of signals * <br>
'32PSK', '16APSK', '32QAM', 'FM', 'GMSK', '32APSK', 'OQPSK', '8ASK', 'BPSK', '8PSK', 'AM-SSB-SC', '4ASK', '16PSK', '64APSK', '128QAM', '128APSK','AM-DSB-SC', 'AM-SSB-WC', '64QAM', 'QPSK', '256QAM', 'AM-DSB-WC', 'OOK', '16QAM'

# Challenges
<img src="https://github.com/alexivaner/Deep-Learning-Based-Radio-Signal-Classification/raw/main/Submission/Final/challenges.jpg" width="800"><br>


# Highlight
This is the improved code from previous project :<br>
Paper ：[Over the Air Deep Learning Based Radio Signal Classification](https://arxiv.org/pdf/1712.04578.pdf)<br>
Github ：[ResNet-for-Radio-Recognition](https://github.com/liuzhejun/ResNet-for-Radio-Recognition)<br>
Previous paper could not recognize signal in  Low SNR value, hence we introduced new method that could also recognize Low SNR signal.


### Proposed Method
<img src="https://github.com/alexivaner/Deep-Learning-Based-Radio-Signal-Classification/raw/main/Submission/Final/Proposed%20Method.jpg" width="700"><br>

### Our Result (Green Line)
We could see that our result surpassed previous method a lot in Low SNR, from under 20% to more than 70% (We could see our result in green line surpassed baseline in Low SNR Signal) <br>
<img src="https://github.com/alexivaner/Deep-Learning-Based-Radio-Signal-Classification/raw/main/Submission/Final/Comparison%202.gif" width="600"><br><br>

#### Comparison in Confussion Matrices:
We could see that we got very good confussion matrices even in the Low SNR Signal
<img src="https://github.com/alexivaner/Deep-Learning-Based-Radio-Signal-Classification/raw/main/Submission/Final/Comparison.gif" width="600"><br>

# How to Run
You could run the program by following steps::<br>
### Clone the repository:<br>
 `git clone https://github.com/alexivaner/Deep-Learning-Based-Radio-Signal-Classification.git` <br><br>
 
### Download the Weight and Dataset:
* Download weights [here](https://drive.google.com/drive/folders/1RIjrZaKJW8oCLbd5LANvTqemk8f-1uWK?usp=sharing) <br>
* Extract all weights to "Submission" folder <br>
* Download extracted dataset [here](https://drive.google.com/file/d/1gUPDlvPqCnb_C4k2h3st0JV9p_sSvaiI/view?usp=sharing)<br>
Our dataset is originally taken from DEEPSIG DATASET: RADIOML 2018.01A (NEW), if you want to know more, you can click [here](https://www.deepsig.ai/datasets)<br>
* Create new folder and name it "ExtractDataset", extract all the dataset and put on that folder, the folder structure will be like below:
  
<pre>
└── <font color="#3465A4"><b>Deep-Learning-Based-Radio-Signal-Classification</b></font>
    ├── <font color="#3465A4"><b>Submission</b></font>
    │   ├── resnet_model_mix.h5
    │   ├── trafo_model.data-00000-of-00001
    │   ├── trafo_model.index
    │   ├── ...
    ├── <font color="#3465A4"><b>Trial</b></font>
    │   ├── ...
    ├── <font color="#3465A4"><b>ExtractDataset</b></font>
    │   ├── part0.h5
    │   ├── part1.h5
    │   ├── part2.h5
    │   ├── ....
</pre>
 
### Inference and Evaluate
* Go to Submission folder: <br>
 `cd Submission` <br>
* Run Jupyter Notebook: <br>
 `jupyter notebook` <br>
* Open "Evaluate-Benchmark.ipnyb": <br>
 
### Training Resnet Modified for High SNR Signal
* Go to Submission folder: <br>
 `cd Submission` <br>
* Run Jupyter Notebook: <br>
 `jupyter notebook` <br>
* Open "Classification-proposed-model-resnet-modified-highest.ipynb": <br>

### Training Transformer Model for Low SNR Signal
* Go to Submission folder: <br>
 `cd Submission` <br>
* Run Jupyter Notebook: <br>
 `jupyter notebook` <br>
* Open "Classification-proposed-model-transformer-low.ipynb": <br>

## Full Proposal
Please download our full proposal here:<br>
[Full Proposal](https://github.com/alexivaner/Deep-Learning-Based-Radio-Signal-Classification/raw/main/Submission/Proposal/Proposal_Team3_Farhan%20Tandia_Ivan%20Surya%20H.pdf)

## Full Final Explanation Report
Please download our full final report here:<br>
[Full Report](https://github.com/alexivaner/Deep-Learning-Based-Radio-Signal-Classification/raw/main/Submission/Final/Final_Team13_Farhan%20Tandia_Ivan%20Surya%20H.pdf)

## Disclaimer
Please cite us as author and our GitHub, if you plan to use this as your next research or any paper.

# Reference
<pre>
T. J. O’Shea, T. Roy and T. C. Clancy, "Over-the-Air Deep Learning Based Radio Signal Classification," in IEEE Journal of Selected Topics in Signal Processing, vol. 12, no. 1, pp. 168-179, Feb. 2018, doi: 10.1109/JSTSP.2018.2797022.<br>

Harper, Clayton A., et al. "Enhanced Automatic Modulation Classification using Deep Convolutional Latent Space Pooling." ASILOMAR Conference on Signals, Systems, and Computers.  2020. <br>

Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones, L.; Gomez, A. N.; Kaiser, Ł. & Polosukhin, I. (2017), Attention is all you need, in 'Advances in Neural Information Processing Systems' , pp. 5998--6008 . <br>

J. Uppal, M. Hegarty, W. Haftel, P. A. Sallee, H. Brown Cribbs and H. H. Huang, "High-Performance Deep Learning Classification for Radio Signals," 2019 53rd Asilomar Conference on Signals, Systems, and Computers, Pacific Grove, CA, USA, 2019, pp. 1026-1029, doi: 10.1109/IEEECONF44664.2019.9048897. <br>

S. Huang et al., "Automatic Modulation Classification Using Compressive Convolutional Neural Network," in IEEE Access, vol. 7, pp. 79636-79643, 2019, DOI: 10.1109/ACCESS.2019.2921988. <br>

Huynh-The, Thien & Hua, Cam-Hao & Pham, Quoc-Viet & Kim, Dong-Seong. (2020). MCNet: An Efficient CNN Architecture for Robust Automatic Modulation Classification. IEEE Communications Letters. 24. 811-815. 10.1109/LCOMM.2020.2968030. <br>
</pre>
