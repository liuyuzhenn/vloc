# Import Matches to Colmap


**Directory organization**

Images should be put in the `img` folder. 
During reconstruction, the following directories will be generated: 
`patches`, `descriptors`, `keypoints`, `matches`, `sparse`; 
the following files will be generated `database.db`, `image-pairs.txt`, `log.txt`, `result.txt`

An example is shown below:

```
data
├─scene1 (root_dir for scene1)
│   └─img
│      ├─img1.jpg
│      ├─img2.jpg
│      └─img3.jpg
├─scene2 (root_dir for scene2)
│   └─img
│      ├─img1.jpg
│      ├─img2.jpg
│      └─img3.jpg
...
```

## Reconstruction

```py
python ./src/reconstruct.py --colmap_path /path/to/colmap --root_dir /path/to/root_dir --device 0
```

Once the reconstruction is done, the structure looks like this (* denotes the generated folders/files): 
```
data
├─scene1 (root_dir for scene1)
│   ├─patches *
│   ├─descriptors *
│   ├─keypoints *
│   ├─matches *
│   ├─sparse *
│   │   └─0 *
│   ├─image-pairs.txt *
│   ├─database.db *
│   ├─log.txt *
│   ├─result.txt *
│   └─img
│      ├─img1.jpg
│      ├─img2.jpg
│      └─img3.jpg
├─scene2 (root_dir for scene2)
│   ├─patches *
│   ├─descriptors *
│   ├─keypoints *
│   ├─matches *
│   ├─sparse *
│   │   └─0 *
│   ├─image-pairs.txt *
│   ├─database.db *
│   ├─log.txt *
│   ├─result.txt *
│   └─img
│      ├─img1.jpg
│      ├─img2.jpg
│      └─img3.jpg
...
```
