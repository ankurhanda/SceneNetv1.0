#SceneNet

All the necessary source code for SceneNet  [SceneNet: Understanding Real World Indoor Scenes with Synthetic Data](http://arxiv.org/abs/1511.07041) will be available here soon.

#Updates

This code enables depth and annotation rendering given a 3D model and trajectory. We provide a sample 3D model in the **data** folder and a trajectory in **data/room_89_simple_data** folder. More details will be updated soon. Things to do 

- [ ] Simulated Annealing
- [ ] [RGB Rendering](https://github.com/ankurhanda/SceneGraphRendering) to be merged with the code
- [ ] Releasing Trajectories
- [ ] Converting Depth to DHA format
- [ ] Emphasise on Negative Focal Length

#Dependencies

* [Pangolin] (https://github.com/ankurhanda/Pangolin-local) Local copy of https://github.com/stevenlovegrove/Pangolin 
* [CVD](https://github.com/ankurhanda/libcvd) (It will not be a dependency any more in future!)
* [TooN](https://github.com/ankurhanda/TooN) (It will be replaced by Eigen3 in future!)
* [ImageUtilities](https://github.com/ankurhanda/imageutilities)
* [SceneGraphRendering](https://github.com/ankurhanda/SceneGraphRendering) (will be merged within soon!)
* OpenCV/OpenCV2
* libnoise (from synaptic)

#Build

```
mkdir build
cd build
cmake .. -DCUDA_PROPAGATE_HOST_FLAGS=0
make -j8
```

#Demo
in your build, run

```
./opengl_depth_rendering ../data/room_89_simple.obj
```
You should have annotated images in the folder **data/room_89_simple_data** that should look like these

![Montage-0](Resources/out.png)

#Adding noise to the depth maps 

[Video](https://youtu.be/3nmQ3SiZKuk?t=42s)


#SceneNet Basis Models (Work In Progress) 

- [ ] [Bedrooms](https://bitbucket.org/robotvault/bedroomscenenet/)
- [ ] [Livingrooms](https://bitbucket.org/robotvault/livingroomsscenenet/)
- [ ] [Offices](https://bitbucket.org/robotvault/officesscenenet)
- [ ] [Kitchens](https://bitbucket.org/robotvault/kitchensscenenet/)
- [ ] [Bathrooms](https://bitbucket.org/robotvault/bathroomscenenet/)

#Website 

More information is available here at our website [robotvault.bitbucket.org](http://robotvault.bitbucket.org)

#Latex Code 

```
\usepackage[table]{xcolor}
\definecolor{bedColor}{rgb}{0, 0, 1}
\definecolor{booksColor}{rgb}{0.9137,0.3490,0.1882}
\definecolor{ceilColor}{rgb}{0, 0.8549, 0}
\definecolor{chairColor}{rgb}{0.5843,0,0.9412}
\definecolor{floorColor}{rgb}{0.8706,0.9451,0.0941}
\definecolor{furnColor}{rgb}{1.0000,0.8078,0.8078}
\definecolor{objsColor}{rgb}{0,0.8784,0.8980}
\definecolor{paintColor}{rgb}{0.4157,0.5333,0.8000}
\definecolor{sofaColor}{rgb}{0.4588,0.1137,0.1608}
\definecolor{tableColor}{rgb}{0.9412,0.1373,0.9216}
\definecolor{tvColor}{rgb}{0,0.6549,0.6118}
\definecolor{wallColor}{rgb}{0.9765,0.5451,0}
\definecolor{windColor}{rgb}{0.8824,0.8980,0.7608}
```

```
\begin{table*}
\begin{tabular}{l}
\textbf{13 class semantic segmentation: NYUv2} \\
\end{tabular}
\centering
\begin{tabular}{|l|p{0.5cm}|p{0.5cm}|p{0.5cm}|p{0.5cm}|p{0.5cm}|p{0.5cm}|p{0.5cm}|p{0.5cm}|p{0.5cm}|p{0.5cm}|p{0.5cm}|p{0.5cm}|p{0.5cm}|p{0.5cm}|}
\hline
Training  & \cellcolor{bedColor}\rotatebox{90}{bed} & \cellcolor{booksColor}\rotatebox{90}{books}  & \cellcolor{ceilColor}\rotatebox{90}{ceil.} & \cellcolor{chairColor}\rotatebox{90}{chair}  & \cellcolor{floorColor}\rotatebox{90}{floor}  & \cellcolor{furnColor}\rotatebox{90}{furn}   & \cellcolor{objsColor}\rotatebox{90}{objs.} & \cellcolor{paintColor}\rotatebox{90}{paint.} & \cellcolor{sofaColor}\rotatebox{90}{sofa}   & \cellcolor{tableColor}\rotatebox{90}{table}  & \cellcolor{tvColor}\rotatebox{90}{tv}     & \cellcolor{wallColor}\rotatebox{90}{wall}   & \cellcolor{windColor}\rotatebox{90}{window} \\ \hline
NYU-DHA & 67.7 & 6.5 & 69.9 & 47.9 & \textbf{96.2} & 53.8 & 46.5 & 11.3 & 50.7 & 41.6 & 10.8 & 85.0 & 25.8 \\ \hline
SceneNet-DHA & 60.8 & 2.0 & 44.2 & 68.3 & 90.2 & 26.4 & 27.6 &  6.3 & 21.1 & 42.2 & 0 & \textbf{92.0} & 0.0 \\ \hline
SceneNet-FT-NYU-DHA & 70.8 & 5.3 & 75.0 & 58.9 & 95.9 & 63.3 & 48.4 & 15.2 & 58.0 & 43.6 & 22.3 &  85.1  & 29.9 \\ \hline
NYU-DO-DHA & 69.6 & 3.1 & 69.3 & 53.2 & 95.9 & 60.0 & 49.0 & 11.6 & 52.7 & 40.2 & 17.3 & 85.0 & 27.1 \\ \hline
SceneNet-DO-DHA & 67.9 & 4.7 & 41.2 & 67.7 & 87.9 & 38.4 & 25.6 &  6.3 & 16.3 & 43.8 & 0 & 88.6 & 1.0 \\ \hline
SceneNet-FT-NYU-DO-DHA & \textbf{70.8} & 5.5 & 76.2 & 59.6 & 95.9 & \textbf{62.3} & \textbf{50.0} & 18.0 & \textbf{61.3} & 42.2 & 22.2 & 86.1 & 32.1 \\ \hline
Eigen \textit{et al.} (rgbd+normals) \cite{Eigen:etal:ICCV2015} & 61.1 & \textbf{49.7} & 78.3 & \textbf{72.1} & 96.0 & 55.1 & 40.7 &\textbf{58.7} & 45.8 &\textbf{44.9}& \textbf{41.9} & 88.7 & \textbf{57.7}  \\ \hline
Hermans \textit{et al.}(rgbd+crf)\cite{Hermans:etal:ICRA2014} & 68.4 & N/A & \textbf{83.4} & 41.9 & 91.5 & 37.1 & 8.6 & N/A & 28.5 & 27.7 & 38.4 & 71.8 & 46.1 \\ \hline
\end{tabular}
\vspace{0.5mm} \vspace{0.5mm}
\caption{Results on NYUv2 test data for 13 semantic classes. We see a similar pattern here --- adding synthetic data helps immensely in improving the performance of nearly all functional categories of objects using DHA as input channels. As expected, accuracy on \textit{books}, \textit{painting}, \textit{tv}, and \textit{windows}, is compromised highlighting that the role of depth as a modality to segment these objects is limited. Note that we recomputed the accuracies of \cite{Eigen:etal:ICCV2015} using their publicly available annotations of 320$\times$240 and resizing them to 224$\times$224. Hermans \textit{et al.} \cite{Hermans:etal:ICRA2014} use ``\textit{Decoration}" and ``\textit{Bookshelf}" instead of \textit{painting} and \textit{books} as the other two classes. Therefore, they are not directly comparable. Also, their annotations are not publicly available but we have still added their results in the table. Note that they use 640$\times$480. Poor performance of SceneNet-DHA and SceneNet-DO-DHA on \textit{tv} and \textit{windows} is mainly due to limited training data for these classes in SceneNet.}
\label{table: CA breakdown for 13 classes}
\end{table*}
```
