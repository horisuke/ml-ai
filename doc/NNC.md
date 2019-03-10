# Sample Project実行
* NNCにログイン
* Project画面を表示し、"tutorial.basics.01_logistic_regression"を選択。
* "You can create a new project from this sample project. Enter a new project name between 1 and 255 characters log."と表示されるので、"tutorial.basics.01_logistic_regression-trial"として、OKを選択。
* 作成したプロジェクトがリストされるので、選択すると、EDIT画面に遷移する。
* EDIT画面には既に作成済みのニューラルネットワークが表示される。
* ここではSmall MNISTを学習データセットとして使用する。
* これはMNISTから一部を抽出したもので、手書き数字0～9が描かれたグレースケール画像の集まりで画像一枚のサイズは28×28px。
* DATASETタブを選択し、Small MNISTを確認できる。
* 学習データ(mnist.small_mnist_4or9_training)及びテストデータ(mnist.small_mnist_4or9_test)はそれぞれ左側のタブで切り替えて確認ができる。
* 各データは以下の要素で構成されている。
    * index：画像番号(連番)
    * x:image：画像データ
    * y:label：o or 1(画像が4の場合は0、9の場合は1)
* また画面上部には学習・テスト時の設定の有無が表示されており、有効にする場合はチェックを入れる。
    * Shuffle：学習時にデータセットをシャッフルする。
    * Image Normalization(1.0/255.0)：ピクセル値を1/225にして値を0.0～1.0にする。
* EDITタブを選択し、作成済みネットワークを表示し、構成を確認する。
* 作成済みネットワークは大きく入力層(I)と出力層(A・S・B)から成る。
* コンポーネントを選択すると、左下のLayer Propertyにレイヤの詳細設定が表示される。
* また各層の右側にはその層の入力サイズが表示されている。
* 次にCONFIGタブを選択し、学習条件を確認する。
* 左側のタブでまず、Global Configを選択し、学習の全体的な条件を設定する
* ～P.82