# 画像読み込みのチュートリアル（機械学習用）

## 実行環境
Ubuntu 22.04
Python 3.12.4

本チュートリアルは，画像学習のおけるデータ読み込みに関する内容である．機械学習が可能なように大量の画像を読み込むことが目的である．ここに書いてあることは，あくまで私の経験に基づいた方法なのでより良い方法があるかもしれないので，ぜひ自分で模索してほしい．

使用するコードは，githubの[load_images_tutorial](https://github.com/uip-kleh/load_images_tutorial)からアクセスできる．

```bash:
git clone https://github.com/uip-kleh/load_images_tutorial
```

実行するだけでは満足せず，しっかり手を動かして理解すること．

## 一枚の画像の読み込み(load_image.py)

classAとclassBの画像をそれぞれ$3$枚ずつ用意した．classAの0.pngを読み込む．
![](classA/0.png)

Pythonには，画像処理が簡単にできるライブラリPillowがあり，これを用いることで用意に画像を読み込むことができる．shapeを確認すると(256， 256， 3)であることがわかる．

```python:
def load_image():
    image = Image.open("classA/0.png")

    return np.array(image)
```
と関数を定義してしまうと，複数の画像を読むこむときに再利用できないので，
```python:
def load_image(fname):
    image = Image.open(fname)

    return np.array(image)
```
と書くことで，ファイル名を引数とすることで読み込むことができる．

## classA内の画像の読み込み

先ほど作成した，一枚の画像を読み込む関数を用いて実装する．同様に再利用できるようにディレクトリ名を引数とする．
```python:
def load_images(dname: str) -> list:
    images = []
    list = glob.glob(dname)
    for fname in list:
        image = load_image(fname)
        images.append(image)
    return images
```

## ImageDataGenerator

この読み込み方法は，使用するデータをメモリに読み込みため少ないデータに対しては有効であるが，機械学習などの大量のデータを用いる場合には，メモリにデータをすべて読み込めない．そこで，学習するミニバッチのみメモリに読み込み，大量のデータを学習する方法がある．これは，実際に自分で実装することも可能であるが，kerasに非常に便利な[ImageDataGenerator](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator)ライブラリがあるためこれを用いる．

- flow_from_directoryメソッド（load_gen_directory.py）

load_gen_directory.pyを実行すると以下のメッセージが表示される.
```bash:
Found 5 images belonging to 2 classes.
```
ディレクトリ名（ここではimages）を引数とすると，以下のディレクトリをラベルとして画像を読み込んでくれる．学習の仕方は各自で確認してほしい．imagesのディレクトリは
- images
    - classA
    - classB

実際にimage_data[0][0]のshapeとラベルを出力すると
```bash:
(6， 256， 256， 3)
[0 0 0 1 1 1]
```
となっている．shapeから$256\times256\times3$の画像が$5$枚読み込まれていることがわかる．また，ラベルはclassAが$0$，classBが$1$と自動で割り当てられている.ImageDataGeneratorは大変便利なように見えるが，学習用と検証用に画像を分割するときはディレクトリを以下のような配置にしなければならない．（私の知っている限り）
- train
    - classA
    - classB
- test
    - classA
    - classB

また読み込む際には，
```python:
# 学習用
train_data_gen = image_gen.flow_from_directory(
        directory="train"  # ディレクトリ名
)
# 検証用
test_data_gen = image_gen.flow_from_directory(
        directory="test"  # ディレクトリ名
)
```

この程度だと苦にならないのだが，交差検証を行なうときには以下のようにディレクトリを配置する．（$2$分割交差検証）
- data0
    - train
        - classA
        - classB
    - test
        - classA
        - classB
- data1
    - train
        - classA
        - classB
    - test
        - classA
        - classB

$2$分割ですらこの様に大変めんどくさい上に，$n$交差検証するときは補助記憶装置をデータは$n$倍に膨らんでしまう．

- flow_from_dataframe

ここで，交差検証を見据えるならこちらを使用したほうが良い．flow_from_directoryではディレクトリ名を引数としていたが，flow_from_dataframeではDataFrameを引数とする．ネット上にはflow_from_directorはかなり散見されるが，こちらはあまり載っていない．

```python:
image_gen = ImageDataGenerator()
image_data = image_gen.flow_from_dataframe(
    dataframe=df， 
    directory="images"，
    x_col="path"，
    y_col="label"
)
```

flow_from_directoryと同じ結果が得られる．DataFrameを引数とするflow_from_dataframeでは，交差検証するのはかなり容易である．これは，DataFrameを分割するだけで学習用と検証用に分割することができ，skcit-learnに実装されているtrain_test_splitやkFoldを使用することができるためである．

## まとめ
ミニバッチ毎に読み込むImageDataGeneratorは便利であり，結論としてはflow_from_dataframeメソッドを使ったほうが良い（交差検証を考慮すると）．また，論文を書いている途中，tensorflowよりもpytorchを使ったほうが良いと思った．tensorflowは細かい実装はほとんど載っていないが，pytorchでは実装が数式などで表現されており引用しやすいためである．