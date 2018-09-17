## TensorFlow/Kerasメモ
* Tensorflowとは
    * Googleが中心に開発しているOSSの機械学習ライブラリ。もともとはGoogle内部で使用するためにGoogle Brainチームによって開発されたもの。
    * Googlr Brainの成果はGoogle検索ランキングやGoogleフォトの画像分類、音声認識などの商用サービスで利用されている。
* 深層学習とは
    * 人間の「脳」のニューロンを模したニューラルネットワークを何層にも重ね、大規模にした機械学習の一手法
    * 2012年のILSVRC(ImageNet Large Scale Visual Recognition Challenge)という画像認識コンペティションで深層学習を採用したチームが圧倒的大差で優勝した。
    * また同じ2012年にGoogleが「深層学習によって教師データなしで猫の概念を自動で学習できた」ことを発表。
    * 2012年に起きたこれらの成果が現在の深層学習・AIブームに繋がっている。
* 深層学習でできること
    * 大きく画像処理、自然言語処理、音声処理、強化学習などを行うことができる。
    * 画像分類
        * 画像に写っているものを推定するタスクのこと。
            * この写真に写っているのは0～9のどの数字か？
            * この顔写真は男性か女性か？
        * ILSVRCではフラミンゴやゴンドラといった1000クラスで分類を行い、その精度を競っていた。
        * いわゆる教師あり学習と呼ばれるもので、事前に多くの画像と正解データが必要となる。ILSVRCでは約1000万枚のデータを用いている。
            * 教師あり学習：事前に与えられた正解データを元に学習する機械学習手法でラベルを推定する分類問題、連続値を推定する回帰問題などがある。
        * 画像収集はWebスクレイピングなどで半自動で行うことが多いが、正解データは人の手で作る必要があり、手間と時間がかかる。
        * そういう背景から、以下に少ないデータで学習するかという研究も活発に行われている。
            * 転移学習
                * ある問題を解くために学習したモデルを流用して別の問題を解くモデルを構築する手法
            * one-shot learning
                * 1つもしくはごく少数のサンプルのみで学習する機械学習の手法
        * 画像分類は各種クラウドサービスがAPIを提供している。機能や精度に違いはあるものの、専門的な知識なしに使えるようになっている。
            * Google Cloud Platform：Cloud Vision API
            * Microsoft Azure：Computer Vision API
            * Amazon AWS：Amazon Rekognition
            * IBM Cloud：Visual Recognition
    * 物体検出
        * 画像分類が「それが何か」を推定するのに対し、物体検出は1つ以上の物体が「何」で画像の「どこ」にあるかを推定すること。
        * 物体検出も各種クラウドサービスでAPIが提供されている。
        * Tensorflowとしては、Tensorflow Object Detection APIが利用可能となっており、画像データを準備すれば、その画像に関する物体検出をすぐに体験できる。
    * セグメンテーション
        * 物体を矩形領域で囲む物体検出に対し、セグメンテーションは物体をピクセル単位で物体ごとに推定すること。
        * 矩形領域を推定する物体検出に比べて、上位の技術のように考えられることもあるが、物体の数を数えたい場合などはセグメンテーションでは物体の重なりにより、うまく数えることができない、などデメリットもある。
    * 画像変換・画風変換
        * ある画像を別の画像に変換すること。
        * 例えば、風景写真を画家が描いた絵のように変換する画風変換も画像変換の1つ。
        * 画風変換は"A Neural Algorithm of Artistic Style"という論文で高速に変換できる手法が提案されている。
        * 他にも"pix2pix"という手法により、線画から写真を生成する手法や、"Cycle GAN"という手法により、夏の景色から冬の景色を生成する手法、白黒写真に自動的に着色する手法などが提案されている。
    * 超解像
        * 低解像画像から高解像画像を生成すること。2015年にwaifu2xというWebサービスも登場している。
        * 超解像も画像変換の一種だが、入力画像と出力画像のサイズが異なる、という特徴がある。
    * 画像生成
        * 画風変換や超解像のような入力データを画像として別の画像を生成するのに対し、ランダムな数値やテキストなどから別の画像を生成すること。
        * GAN(Generative Adversarial Network)と呼ばれる手法を元に画像生成する研究が進められている。
            * 多段の隠れ層と畳み込み層を持つネットワークを持つGANをDCGAN(Deep Convolutional Generative Adversarial Network)と呼ぶ。
        * 多くの改善手法も多数発表されており、リアルな画像生成ができるようになってきている。
        * GANは文章や時系列データの生成にGANを利用する研究も進められているため、画像生成にとどまらず、さまざまなタスクに利用されていく可能性がある。
    * 文章分類
        * その文章がどのカテゴリに属するかを推定するタスクのこと。
        * 有名な応用例としては、スパムメールの自動判定などが挙げられる。
        * 文書分類は様々なタスクの基礎となるタスクのため、自然言語処理の様々なアルゴリズムで利用されている。
            * 対話システムではユーザからの入力文の意図(Intent)を抽出する必要があるが、そこでも文書分類が使用されている。
            * GoogleのSmart ReplyではSmart Replyの対象にするかどうかに関して文書分類が使用されている。
    * 対話文生成
        * 画像生成と同様、文章を生成する。
        * 2015年の"A Neural Conversational Model"という論文では、映画の字幕データやIT系ヘルプデスクのやり取りのデータを使って、自然な会話文を生成できることを示した。
    * 機械翻訳
        * 機械翻訳の歴史は古く、当時はルールベースの手法だった。その後、統計的な手法を経て、現在では深層学習が使われるようになっている。
        * 2016年11月のGoogle翻訳のアップデートで英日翻訳の精度が劇的に改善したが、このアップデートで翻訳アルゴリズムが深層学習を用いた手法(GNMT：Google's Neural Machine Translation)に置き換わったからだとされている。
        * GNMTは詳細が公開されており、そのベースは対話文生成で用いられている技術と同じである。
    * 文書要約
        * 自然言語処理の主要な分野の1つだが、文書要約には様々な技術がある。
            * 単一文書要約：1つの長い文章を短くする
            * 複数文書要約：Twitterなどの短い文章が大量にある場合、それらを代表する文章(tweet)を抽出する
        * 上記のような抽出的アプローチだけでなく、必ずしも文章に含まれているとは限らない単語を使った文を生成する「生成的アプローチ」も研究が進んでいる。
            * 2016年にGoogle Brainチームが「ニュース冒頭文から良い見出しを生成するアルゴリズム」を発表
            * ニュースの冒頭の文章から見出しを生成し、人間的な要約を実現。
    * 対話システム
        * レストランのレコメンドなど特定の目的の達成を目指す「タスク指向」の対話システムとMicrosoftのりんなに代表される、特定の目的を持たない「非タスク指向」の対話システムがある。
        * ただ対話システムの実際の現場では、深層学習を用いた高度な手法よりも、コントロールしやすいルールベースの手法がまだ主流である。
        * 実用的な対話システムを深層学習で実現するためには、様々な要素を組み合わせる必要があり、単一の深層学習アルゴリズムだけでは人間的な対話がまだできるようにはなっていない。
    * 音声認識
        * 画像処理や自然言語処理と並び、音声認識も深層学習の主要な応用分野である。
        * 実アプリケーションでも深層学習の利用がされており、iPhoneやAndroid端末の音声認識の精度向上に繋がっている。
    * 音声合成・音楽生成
        * 2013年以降、音声合成や音楽生成の分野でも深層学習が利用され始めている。
            * Google Brainチームの取り組みの1つに深層学習をアートに適用する「Project Magenta」がある。
            * Project Magentaの成果物であるPerformance RNNを使うと、単純なMIDI信号からプロが演奏したような出力を得ることができる。
            * Google DeepMindによるWaveNetも2016年に注目を集めた。これまでの音楽生成は出力がMIDIの楽譜の場合が多かったが、WaveNetでは直接波形を生成し、質の高い音楽生成を実現している。
    * 声質変換
        * 普通の音声合成はテキストから声を生成するのに対し、声色だけを変換する技術、いわゆるボイスチェンジャーの発展系。
        * 声質変換技術を使うことで自分の話している内容を別の人の声に変換することができる。
    * ゲームの攻略
        * 2015年、DeepMindが開発したDQNがAtari2600というゲームにおいて、人間のゲーマー同等の強さを示した。
            * DQNはDeep Q Networkの略で深層学習と強化学習の一種Q Learningを組み合わせた手法。これまでは深層学習と強化学習の組み合わせで学習を安定させることができなかったが、様々な工夫により、安定的な学習を可能にした初めての例である。
        * 2016年には人間には勝てないと言われていた囲碁でDeepMindが開発したAlphaGoがトップ棋士を圧倒した。
        * DQNもAlphaGoも深層学習と強化学習を組み合わせた深層強化学習と呼ばれる手法である。
            * AlphaGoはDeepMind開発の囲碁プログラムで様々なバージョンがある。
                * AlphaGo Lee：トップ棋士に勝利したプログラム
                * AlphaGo Master：ネット碁でプロ棋士に60連勝を達成したプログラム
                * AlphaGo Zero：人間の棋譜を全く学習せず、3日間の学習でAlphaGo Lee/Masterに圧勝したプログラム
                * Alpha Zero：AlphaGo Zeroを将棋やチェスにも対応可能にしたプログラム
    * システムの制御
        * Preferred Networksは深層強化学習を使ってロボットカーが障害物を自動で避ける動作をゼロから学習するデモを公開。
        * DeepMindは"Deep Reinforcement Learning for Robotic Ⅳlanipulation with Asynchronous(Dff Policy Updates"でロボットアームにドアの開け方を自動学習させている。
    * 時系列データの予測・分類
        * 自然言語や音声データに限らず、深層学習は様々な時系列データに応用できる。
            * 株価予測
            * 行動推定(人間の体につけたジャイロセンサーのデータから歩いている、階段上がっているといった行動を推定する)
    * 異常検知
        * 他の多くのデータとは振る舞いが異なるデータを検出する技術のこと。
            * クレジットカードの不正検知、システムの故障検知、不良品の検出など
        * 異常検知自体は以前から行われていたが、深層学習により、精度の向上、対象データの拡大が行える。
* TensorFlowの特徴
    * 有向非巡回グラフ(DAG：Directed acyclic graph)
        * TensorFlowは有向非巡回グラフをベースとした処理系である。
        * Tensor(テンソル)はベクトルや行列を一般化した概念。
        * テンソル同士の演算結果はテンソルになるため、複雑な演算であっても、お互いに矢印で結ばれたループの内ネットワーク(=有向非巡回グラフ)で表現できる。
        * このネットワークは計算グラフとも呼ばれる。
        * 減速として、TensorFlowはPythonでこの計算グラフの定義を行い、その定義が完了してから複雑な処理をまとめて一気に行うことで高速な演算を実現している。
        * 加えて、グラフの実行は処理の速いC++で行う。また一気に計算することでその計算結果をPython側に転送するのもコストが小さくて済む。
    * いろいろな環境で動作
        * 基本的にCPUでもGPUでも同じコードを動かすことが可能。
        * Pythonを使って定義し、構築したグラフを保存し、それを別言語から呼び出すことも可能。
        * つまり、iPhoneやAndroid端末でも構築したグラフを動作させることができる。
        * "TensorFlow Lite"では、深層学習のモデルが通常、数百MBにもなってしまう場合にそのモデルを圧縮し、より制約のある端末でもモデルを使用することができる。
        * "TensorFlow.js"では、JavaScriptでTensorFlowのモデルを高速に実行できる。
        * "TensorFlow Serving"では、構築したモデルを簡単にAPIとして提供できる。
    * 分散処理
        * TensorFlowのホワイトペーパーではその分散処理の仕組みを中心に紹介されており、TensorFlowの大きな強みと言える。
        * 深層学習は計算量が非常に大きく、分散処理が必要になるケースが多くある。
        * 一般には分散処理を使いこなすのは非常に高度な技術力が必要となるが、TensorFlowを用いることでかんたんに分散処理を記述できる。
    * TensorBoardによる可視化
        * 深層学習はブラックボックスと言われるが、その表現力の高さゆえに何が起きているかわかりづらい、という問題がある。
        * TenaorFlowに付属するTensorBoardというツールは、学習時の損失関数の経過や中間層の様子、抽出した特徴量の埋め込みによる可視化などができ、今何が起きているかを理解する助けになる機能が備わっている。
        * 可視化した結果からデバッグや構築したモデルを理解したりできる。
    * 様々なレベルのAPIとエコシステム
        * 細かい制御が可能な低レベルAPIから高レベルAPIまで幅広くカバーしている。
        * 最新のver.ではCore APIの上にLayersやKeras、Estimatorなどの高レベルなAPIが提供されている。
        * pre-made EstimatorはEstimatorに含まれる基本的なネットワーク構造が決まっていて、パラメータを設定するだけですぐに学習を始められる仕組みで、TensorFlowの敷居がかなり下がる。
        * pre-made Estimatorで対応できないネットワークはLayerやKerasを用いて、レゴブロックのようにモデルを組み合わせて構築できる。
        * 2017年3月にはGoogle Cloud Machine Learning Engine(MLEngine)が公開され、インフラを準備しなくても、CPU/GPUを利用した分散学習が可能で、構築したモデルをそのままAPIとして利用できる。
        * TensorFLowはコミュニティが非常に大きく、他の深層学習ライブラリを圧倒している。最新の研究論文コードはTensorFlowで実装されていることも多く、コードを読んで理解したり、試すことが容易となっている。
* Kerasとは
    * Francois Choletが中心となって開発している深層学習ライブラリ、もしくはAPI仕様のこと。
    * Kerasには2つの実装がある。1つはTensorFlowに統合されたもので、もう1つはバックエンドとして、TensorFlowに加え、TheanoやCNTKもサポートしている、独立したパッケージである。
    * 前者は「KerasとはAPI使用である」と再定義して実装をTensorFlowに統合している。
    * 後者はこれまで通り、複数のバックエンドを選べる形態を維持している。
    * 両者には若干の違いはあるものの、API仕様が統一されており、どちらも同じように利用できる。
    * モジュール構成がシンプルなのがKerasの特徴の1つ。
    * 深層学習のネットワークを構築する上でよく使われるものが適切な粒度でモジュール化されている。
    * それにより、レゴブロックを組み合わせるような感覚で深層学習モデルを構築できる。
* Define and Run/Define by Run
    * Define and RunはTensorFlowのように先に計算グラフを定義し、まとめて処理する概念である。
    * 通常のプログラミング言語とパラダイムが異なるため、学習コストが若干高い反面、高速化が容易である。
    * Define by RunはPreferred NetworksのChainer開発チームが初めて提唱した概念である。
    * 予め計算グラフを定義せず、グラフ定義と処理を同時に行う手法である。
    * 処理結果によって計算グラフを動的に変えられるため、実装がシンプルになり、デバッグ時にエラー箇所を把握しやすい。
    * 2016年までは高速化の観点からDefine and Runが主流だったが、2017年ごろからはChainer以外でDefine by Run型のライブラリが多く登場している。
    * TensorFlowもv1.5で導入されたEagar Executionを用いることでDefine by Runによる記述が可能になっている。
* 複数ライブラリ間のモデルの共有
    * これまではある深層学習ライブラリで実装したモデルはそのライブラリでしか使用できなかった。
    * 1度学習したモデルのパラメータを別のライブラリでも利用したいというニーズから、「モデルをどう共有するか」に着目したライブラリや使用が登場している。
        * ONNX(Open Neural Network Exchange：深層学習モデルを表現するための共通フォーマット)を用いると、ライブラリAで実装したモデルを読み込み、ライブラリBのモデルに変換することができる。
* 深層学習のエコシステム
    * 2016年ごろからAIや深層学習に関する認知が一般企業に広まってきた。
    * これまでの研究や実験レベルの内容から実サービスでどう活用するか、という話に徐々にシフトしている。
    * 研究ではモデルそのものが重要である一方、実サービスではモデルの監視やアップデート、外部システムとの連携など検討事項が多岐にわたる。
    * 近年では実サービスでの活用にあたっての煩わしさを吸収するサービスをクラウド事業者が提供し始めている。
        * GCPではDataFlowやDataLab、前述のML Engineを駆使することで、データの蓄積から前処理、モデルの構築とサービスのホスティングまでを一貫してクラウド上で実行することができる。
        * 2017年に公開されたColaboratoryは12時間の制約があるものの、GPUを利用して深層学習モデルの構築を行うことができ、GoogleDriveでコードをシェアすることができるようになっている。
        * AWSではSageMakerと呼ばれるフルマネージドなサービスを提供しており、Jupyter Notebookを使ったモデルの構築や学習、モデルのホスティングまでの一連の流れを簡単に行うことができる。
    * これらクラウド機能を利用することで、実サービスでの運用を見据えた形で深層学習に取り組める環境が整いつつある。
* 開発環境構築
    * Pythonの開発環境 Anaconda：省略
    * 動作確認環境 Jupyter Notebook：省略
* TensorFLowの基本
    * データフローグラフ
        * 以下はtensroflowのデータフローグラフの概念を用い、定数a, bにそれぞれ1, 2を代入して足し算をするコードである。
        ```python
        // tensorflow-basic1.py
        import tensorflow as tf

        a = tf.constant(1, name='a')
        b = tf.constant(2, name='b')
        c = a + b
        graph = tf.get_default_graph()

        with tf.Session() as sess:
            print(c)                     // Tensor("add:0", shape=(), dtype=int32)
            print(sess.run(c))           // 3
            print(graph.as_graph_def())  // 省略
        ```
        * 上記のうち、 $1+2=3$ の計算は ```sess.run(c)``` の箇所で行われる。
        * ```c = a + b``` の部分では「aとbの加算結果を値に持つcというTensor」の定義をしているに過ぎない。
        * cは上記の通り、Tensor型インスタンスである。
        * Tensorflowはこのように2つのステップを踏んで実行する。
            1. どのような計算をするかを定義する
            2. まとめて計算を実行する
        * データフローグラフはデータの流れをグラフ(ネットワーク)として表現したもので、計算内容を定義する役割を果たす。
        * グラフは点と点を線でつないだ構造になっており、点のことをノードや頂点、線のことをエッジや辺と呼ぶ。
        * 上述のコードでは、aやbといった定数、加算(add)といった操作は「ノード」に対応する。
        * また、a,bはaddとそれぞれ関係性として結ばれ、「エッジ」に対応する。
        * ```get_default_graph()``` でデータフローグラフの定義を取得し、```as_graph_def()``` でグラフを表示し確認できる。
        * 上記のグラフの定義は下記の通り。
        ```
        node {
          name: "a"
          op: "Const"
          attr {
            key: "dtype"
            value {
              type: DT_INT32
            }
          }
          attr {
            key: "value"
            value {
              tensor {
                dtype: DT_INT32
                tensor_shape {
                }
                int_val: 1
              }
            }
          }
        }
        node {
          name: "b"
          op: "Const"
          attr {
            key: "dtype"
            value {
              type: DT_INT32
            }
          }
          attr {
            key: "value"
            value {
              tensor {
                dtype: DT_INT32
                tensor_shape {
                }
                int_val: 2
              }
            }
          }
        }
        node {
          name: "add"
          op: "Add"
          input: "a"
          input: "b"
          attr {
            key: "T"
            value {
              type: DT_INT32
            }
          }
        }
        versions {
          producer: 26
        }
        ```
    * セッション
        * グラフの計算結果を得るためにはtf.Session()クラスのインスタンスを作成する必要がある
        * 生成したインスタンスでrunメソッドを呼び出し、そこに計算したいノードを指定することで、実行結果を得ることができる。
        * runメソッドには ```run([a, b])``` のように指定することで複数のノードを指定して同時に県産することもできる。
    * データフローグラフの構成要素
        * 上記のコードでは、定数や加算がノードに対応していたが、これらも合わせグラフの構成要素としては、以下がある。
            * 定数
                * ```tf.constant()``` を使って定義できる。1度定義したら、その後変更はできない。
            * 変数
                * ```tf.Variable()``` を使って定義できる。変数は値を変更できるため、学習対象のパラメータを定義しておくことでパラメータの更新、つまり学習が可能となる。
                ```python
                // tensorflow-basic2.py
                import tensorflow as tf

                a = tf.Variable(1, name='a')
                b = tf.constant(1, name='b')
                c = tf.assign(a, a + b)

                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    print('1: [c, a] =', sess.run([c, a]))     // 1: [c, a] = [2, 2]
                    print('2: [c, a] =', sess.run([c, a]))     // 2: [c, a] = [3, 3]
                ```
                * ```tf.assign()``` は変数に値を代入する操作を行なう。上記の場合、「変数aにa+bの結果を代入し、代入後のaを返す」という操作を行なう。
                * 最初の ```sess.run()``` では、a=1の状態から、 cを計算する過程で.
                a=2となり、cも2となる。よって、この時点での出力はa,c共に2となる。
                * 2回目の ```sess.run()``` では、a=2の状態から、 cを計算する過程で.
                a=3となり、cも3となる。よって、この時点での出力はa,c共に3となる。
                * 変数は必ず最初に初期化する必要があり、 ```tf.global_variables_initializer()``` を使用することで全ての変数を初期化できる。
                * 変数を指定して初期化するには、```tf.initialize_variables()``` を使用する。
            * プレースホルダー
                * ```tf.placeholder()``` を使って定義できる。様々な値を受け付けることができる箱のようなもの。
                * 値が未定な状態でグラフを構築し、実行時に具体的な値を指定することが可能なため、主に入力データを扱う部分で利用される。
                ```python
                // tensorflow-basic3.py
                import tensorflow as tf

                a = tf.placeholder(dtype=tf.int32, name='a')
                b = tf.constant(1, name='b')
                c = a + b

                with tf.Session() as sess:
                    print('a + b =', sess.run(c, feed_dict={a: 1}))    // a + b = 2
                ```
                * aの定義部分ではplaceholderで箱を作っているが、値はセットしていない。
                * aの値はrunメソッド実行時にfeed_dictメソッドでaに1をセットしている。feed_dict()は辞書型で引数を指定できるため、複数のplacefolderに対して、実行時に値をセットすることができる。
                * 上記の実行結果は定数b=1と実行時にセットされるa=1の加算となり、　"a + b = 2"となる。
            * 演算
                * 加算・乗算といった演算もグラフのノードして表現される。
                ```python
                // tensorflow-basic4.py
                import tensorflow as tf

                a = tf.constant(2, name='a')
                b = tf.constant(3, name='b')
                c = tf.add(a, b)
                d = tf.multiply(a, b)

                with tf.Session() as sess:
                    print('a + b = ', sess.run(c))    // a + b = 5
                    print('a * b = ', sess.run(d))    // a * b = 6
                ```
    * 多次元配列とテンソル
        * Tensorflowではスカラだけではなく、ベクトルや行列のような多次元配列のデータも扱うことができる。
        * 配列ではない数値をスカラ、スカラやベクトル、行列を含む多次元配列をテンソルと呼ぶ。
            |名称|次元|具体例|表記例|
            |---|---|---|---|
            |スカラ|0|1|$x$|
            |ベクトル|1|[1,2,3]|$x_{i}$|
            |行列|2|[[1,2],[3,4]]|$x_{ij}$|
            |テンソル|任意|[[[1,2],[3,4]],...]|$x_{i...j}$|
        * ベクトル演算
            ```python
            // tensorflow-basic5.py
            import tensorflow as tf

            a = tf.constant([1,2,3], name='a')
            b = tf.constant([4,5,6], name='b')
            c = a + b

            with tf.Session() as sess:
                print('a + b = ', sess.run(c))    // a + b = [5 7 9]
            ```
        * 行列演算
            ```python
            // tensorflow-basic6.py
            import tensorflow as tf

            a = tf.constant([[1,2],[3,4]], name='a')
            b = tf.constant([[1],[2]], name='b')
            c = tf.matmul(a, b)

            print('shape of a:', a.shape)
            print('shape of b:', b.shape)
            print('shape of c:', c.shape)

            with tf.Session() as sess:
                print('a = \n', sess.run(a))
                print('b = \n', sess.run(b))
                print('c = \n', sess.run(c))
            ```
            * 3次元以上の配列でも同様に計算が可能であり、ベクトル演算、行列演算も含め、多次元配列同士の演算をテンソル演算と呼ぶ。
        * テンソル演算とプレースホルダー
            * ```tf.placeholder()``` でスカラ値だけでなく、テンソルを受けられるようにするためには、shape引数で次元を指定する必要がある。
            * テンソルの大きさが決まっていない場合、未知の次元方向について、Noneを指定する。
                ```python
                // tensorflow-basic7.py
                import tensorflow as tf

                a = tf.placeholder(shape=(None, 2), dtype=tf.int32, name='a')

                with tf.Session() as sess:
                    print('-- Insert [[1, 2]] --')
                    print('a = ', sess.run(a, feed_dict={a:[[1,2]]}))           //a =  [[1 2]]
                    print('\n-- Insert [[1, 2], [3, 4]] --')
                    print('a = ', sess.run(a, feed_dict={a: [[1, 2], [3, 4]]})) //a =  [[1 2] [3 4]]
                ```
            * 上記のコードでは、shapeで(None, 2)をしている。これにより、runメソッド呼び出し時に行方向に任意の成分を持つテンソルを渡すことができる。
            * 1番目のrunでは1×2ベクトル、2番目のrunでは2×2行列を渡すことができている。
    * セッションとSaver
        * 前述の通り、変数はセッションごとに初期化が必要である。これは、あるセッションで変数を更新しても、別のセッションではその変数の値は引き継がれないことを意味する。
        * つまり、学習対象のパラメータを変数として定義すると、同一セッションを維持している間でしか、更新後の変数、つまり学習後の更新済みの変数を利用できないことになる。
            ```python
            // tensorflow-basic8.py
            import tensorflow as tf

            a = tf.Variable(1, name='a')
            b = tf.assign(a, a+1)

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                print('1st b = ', sess.run(b))      //1st b =  2
                print('2nd b = ', sess.run(b))      //2nd b =  3

            # session is changed.
            with tf.Session() as sess:
                print('-- New session --')
                sess.run(tf.global_variables_initializer())
                print('1st b = ', sess.run(b))      //1st b =  2
                print('2nd b = ', sess.run(b))      //2nd b =  3
            ```
        * Saverを利用すると、変数の値をファイルに書き出したり、ファイルからプログラムに値を読み込んだりできる。
        * これにより、機械学習モデルを保存したり、別のプロセスで使用することができる。
            ```python
            // tensorflow-basic9.py
            import tensorflow as tf

            a = tf.Variable(1, name='a')
            b = tf.assign(a, a+1)

            saver = tf.train.Saver()
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                print(sess.run(b))                    // 2
                print(sess.run(b))                    // 3
                # Save the values of variables to model/model.ckpt.
                saver.save(sess, 'model/model.ckpt')

            saver = tf.train.Saver()
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                # Restore the values of variables from model/model.ckpt.
                saver.restore(sess, save_path='model/model.ckpt')
                print(sess.run(b))                    // 4
                print(sess.run(b))                    // 5
            ```
    * TensorBoardによるグラフの可視化
        * TensorBoardはTensorflowに付属するツールでモデルの構造や学習状況を可視化できる。
        * グラフ以外にも損失の履歴やベクトルの埋め込みなど様々なものを可視化できる。
        * グラフを可視化するには、以下のように ```tf.summary.FileWriter()``` を使い、必要な情報を書き出す必要がある。
            ```python
            // tensorflow-basic10.py
            import tensorflow as tf

            LOG_DIR = './logs'

            a = tf.constant(1, name='a')
            b = tf.constant(1, name='b')
            c = a + b

            graph = tf.get_default_graph()
            with tf.summary.FileWriter(LOG_DIR) as writer:
                writer.add_graph(graph)
            ```
        * 上記を実行すると、正常終了し、ml-trial/logs以下にグラフの可視化に必要な情報が書き出された"events.out.tfevents.1536461582.ubuntu"のようなファイルが生成される。
        * このファイルをtensorboardで開き、確認する。
        * ml-trialディレクトリ以下に移動し、以下のコマンドでAnacondaの仮想環境に入る。
            * $ source activate ml-trial
        * プロンプトの先頭に(ml-trial)が付き、仮想環境には入れていることを確認する。
        * 以下のコマンドで生成したファイルを指定して、tensorboardを起動する。
            * $ tensorboard --logdir=logs
            * TensorBoard 1.10.0 at http://ubuntu:6006 (Press CTRL+C to quit)
        * 上記のようなURLが表示されるので、ブラウザでアクセスすると、tensorboardが表示される。
        * ブラウザを閉じ、terminalでctrl_Cすることで、tensorboardを閉じる。
        * 以下のコマンドで仮想環境から抜ける
            * $ source deactivate
    * 最適化と勾配法
        * 機械学習・深層学習における学習とは予測の誤差を最小化・最適化することを指す。
        * ここでの最適化は与えられた関数を最小もしくは最大にするようなパラメータを見つけることを意味する。
        * TensorFlowには「勾配法」と呼ばれる手法により、関数を最小化するための機能が備わっている。
        * 勾配法は最適化問題において関数の勾配に関する情報を解の探索に用いるアルゴリズムの総称である。
        * それらのうち、最もシンプルな手法が最急降下法で、以下の手順を辿る。
            1. パラメータを適当な値で初期化
            2. 与えられたパラメータにおける関数の傾き(勾配)を計算
            3. 最も傾きの大きい方向にパラメータを少しずらす
            4. 手順2, 3を繰り返す
        * 最急降下法をゼロから実行しようとすると、特に手順2の勾配の計算部分が難しいが、TensorFlowでは手順2,3を行うための便利な仕組みが用意されている。
        * 2次関数 $y=(x-1)^2$ を最小化する $x$ を見つけるための実装は以下の通り。
            ```python
            // tensorflow-basic11.py
            import tensorflow as tf

            # define parameters as variables.
            x = tf.Variable(0., name='x')
            # define the function to be minimized by parameters.
            func = (x - 1) ** 2

            # set the learning-rate
            optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=0.1
            )

            # train_step is the operation to move the values of x.
            train_step = optimizer.minimize(func)

            # execute train_step repeatly.
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                for i in range(20):
                    sess.run(train_step)
                print('x = ', sess.run(x))                  // x = 0.98847073
            ```
        * 始めにパラメータxを変数として定義し、 xを使って最小化したい関数funcを定義する。
        * ```tf.train.GradientDescentOptimizer``` が最急降下法におけるパラメータの更新を担当し、```minimize()``` メソッドの引数でfuncを渡すことにより、パラメータxを少しずつずらす操作```train_step``` を得ることができる。
        * 「少しずつずらす」は```learning_rate``` で指定する。
        * 最後にforループで```train_step``` を繰り返し実行することで関数funcの最小化処理を行う。
        * この例では初期値を $x=0$ としているが、```train_step``` を20回繰り返すことで最適な値 $x=1$ に近い値 $x =0.98847073$ を得ることができる
        * 勾配法の機械学習への適用
            * 勾配法を機械学習に適用するためにBoston house-pricesデータセットを使用する
            * Boston house-pricesは住宅の部屋数や高速道路へのアクセスのしやすさなど13の変数(説明変数)とそれに対応する住宅価格(中央値)が506個分含まれるデータセットである。
            * このデータセットを使用し、13の変数を受け取り、住宅価格の推定値を出力する関数を学習することを考える。
            * この関数を機械学習モデルと呼ぶ。
                ```python
                // tensorflow-basic12.py
                import tensorflow as tf
                import matplotlib.pyplot as plt

                # Download the dataset of Boston house-prices
                (x_train, y_train), (x_test, y_test) = tf.keras.datasets.boston_housing.load_data()

                # global settings of matplotlib
                plt.rcParams['font.size'] = 10*3
                plt.rcParams['figure.figsize'] = [18, 12]
                plt.rcParams['font.family'] = ['IPAexGothic']

                # display histogram "house prices/$1000" vs "number of data"
                plt.hist(y_train, bins=20)        // bins:number of bar in histogram
                plt.xlabel('house prices/$1000')
                plt.ylabel('number of data')
                plt.show()

                # display plot "number of rooms" vs "house prices/$1000"
                plt.plot(x_train[:, 5], y_train, 'o')
                    // 6th column of x_train is "number of rooms"
                    // 'o' means circle marker
                plt.xlabel('number of rooms')
                plt.ylabel('house prices/$1000')
                plt.show()

                # preprocessing
                x_train_mean = x_train.mean(axis=0)
                x_train_std = x_train.std(axis=0)
                y_train_mean = y_train.mean()
                y_train_std = y_train.std()

                x_train = (x_train - x_train_mean) / x_train_std
                y_train = (y_train - y_train_mean) / y_train_std
                x_test = (x_test - x_train_mean) / x_train_std
                y_test = (y_test - y_train_mean) / y_train_std

                # display plot "number of rooms" vs "house prices/$1000" after preprocessing
                plt.plot(x_train[:, 5], y_train, 'o')
                plt.xlabel('number of rooms(after)')
                plt.ylabel('house prices/$1000(after)')
                plt.show()

                # define the inference model of Boston house-prices
                x = tf.placeholder(tf.float32, (None, 13), name='x')
                y = tf.placeholder(tf.float32, (None, 1), name='y')
                w = tf.Variable(tf.random_normal((13, 1)))
                pred = tf.matmul(x, w)

                # define the loss function and learning rate
                loss = tf.reduce_mean((y-pred)**2)
                optimizer = tf.train.GradientDescentOptimizer(
                    learning_rate=0.1
                )
                train_step = optimizer.minimize(loss)

                # execute train step and repeatedly
                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    for step in range(100):
                        train_loss, _ = sess.run(
                            [loss, train_step],
                            feed_dict={
                                x : x_train,
                                y : y_train.reshape((-1, 1))
                            }
                        )
                        print('step: {}, train_loss: {}'.format(step, train_loss))

                    pred = sess.run(
                        pred,
                        feed_dict={
                            x: x_test
                        }
                    )
                ```
            * まず、```tf.keras.datasets.boston_housing.load_data()```でBoston house-pricesのデータセットをダウンロードする。
                * ダウンロードしたデータは"~/.keras/datasets/"以下に保存される。
            * 返り値は```x_train```, ```y_train```を要素に持つタプル(学習データ：学習に用いるデータ)、及びは```x_test```, ```y_test```を要素に持つタプル(テストデータ：精度の評価用に用いるデータ)となる。
            * ```x_train```, ```x_test```には行方向に各データ、列方向に各説明変数の値が格納されている。
            * ```y_train```, ```y_test```には行方向に各データの住宅価格が格納されている(列方向は1列のみ)。
            * 取得したデータの概要を知るために```y_train```をヒストグラムで表示する。表示のためにはmatplotlibを使用する。
            * ```plt.rcParams()```でmatplotlibのglobalな設定を行う。
            * ```plt.hist()```に```y_train```を渡し、```plt.show()```でヒストグラムを表示する。
            * 横軸は```y_train```の値(=住宅価格)全体を20分割(binsで20を指定)し、棒グラフとして表示する。
            * 縦軸は```y_train```のデータを横軸の各棒グラフに属する数を表す。
            * 次に部屋数と住宅価格の関係をplotすることを考える。
            * ```plt.plot()```に横軸、縦軸に設定するデータを渡し、plotのmarker種を設定する。
                * 部屋数はx_trainの6列目に格納されているので、横軸のデータとして、```x_train[:, 5]```を指定する。
                * 縦軸のデータとして、```y_train```(住宅価格)を指定する。
                * plotのmarkerとして、```'o'```を指定し、circle markerとする。
            * 同様に```plt.show()```でplotを表示する。
            * 入力の前処理(preprocessing)では、まず```mean()```メソッドと```std()```メソッドでそれぞれ平均値、標準偏差を求める。
                * ```x_train```は行方向と列方向にデータが存在するため、行方向に対して、平均値・標準偏差を求めるためには```axis```で0を指定する。
            * これらの値から、(各要素 - 平均値) / 標準偏差を算出し、入力データの標準化を行う。
            * 標準化されたデータ群は平均0、分散1のデータ群となり、データが全体的に原点付近に集まるため、学習やパラメータの調整がしやすくなる。
            * ```x_test```, ```y_test```に対しても標準化を行うが、その際に使用する平均値と標準偏差はそれぞれ、```x_train```, ```y_train```で算出したものを使用する。
            * 次にモデルの定義を行なう。ここではシンプルに各説明変数(x)を重み付き(w)で足し合わせたものモデルとしている。
            * 説明変数(x)と推定値(y)はそれぞれ、(None, 13), (None, 1)のplaceholderとして定義し、重み(w)は(13, 1)の1未満のランダムな値を初期値として持つ変数として設定する。
            * ここでは、重みwがパラメータでpredが予測結果を表すテンソル(モデル)となる。
            * 次に損失関数(=目的関数：最小化したい関数)を定義する。ここでは実測値(y)と推定値(pred)の差の二乗の平均(=最小二乗誤差：MSE/Mean Squared Error)を損失関数としている。
            * 前述の実装と同様、```tf.train.GradientDescentOptimizer``` が最急降下法におけるパラメータの更新を担当するため、```minimize()``` メソッドの引数でlossを渡すことにより、パラメータwを少しずつずらす操作```train_step``` を得ることができる。```learning_rate```は0.1と設定する。
            * 最後に```train_step```を使って、forループを回し、学習を行う。
            * ```tf.Session()```の```run()```メソッドの第1引数```fetches```には、学習した処理結果を取り出したい(=fetchしたい)ネットワーク(=学習したいネットワーク)のインスタンスを設定する。
                * ここでは、損失関数の値```loss```と最急降下法の操作```train_step```をリストとして渡す。
            * 引数```feed_dict```には、定義の時点で不定としていた、x, yに値をfeedする(=書き込む)。
            * ここでは、xに```x_train```、yに```y_train```を設定する。
            * ```run()```メソッドreturn値は```loss```, ```train_step```のそれぞれのreturn値となる。
            * ```loss```は値となるため、```train_loss```として値を保持する。
            * ```train_step```は操作を表すインスタンスなので、return値は存在しない(Noneとなる)。
            * モデルの学習が終了したら、predを```run```メソッドに渡し、x_testをfeed_dictにセットして、評価データを使用し、学習したモデルを評価する。
    * 確率的勾配降下法とミニバッチ
        * 上記では最急降下法を使用して、機械学習のモデルの学習を行ったが、これはデータの数が506と小さなものだったため、全てのデータをメモリ上に1度に展開でき、学習ができた。
        * 実際には数万～数百万のデータを扱うことがあり、そういう場合は確率的勾配降下法(Stochastic Gradient Descent)を使用する。
        * SGDでは全てのデータを1度に使用せず、ミニバッチと呼ばれるデータの塊に分割して学習する。
        * これにより、データが大量の場合にも学習が可能となるのはもちろん、学習の挙動が確率的になるため、局所解に陥りづらくなるといったメリットもある。
        * よって、学習データがそれほど大量でなくても、SGDを使用するのが一般的となっている。
        * 以下はSGDによる学習の実装コードとなる。
            ```python
            // tensorflow-basic13.py
            import tensorflow as tf
            import matplotlib.pyplot as plt
            import numpy as np

            BATCH_SIZE = 32

            def get_batches(x, y, batch_size):
                n_data = len(x)              // リストの長さを取得する
                indices = np.arange(n_data)  // 0～n_data-1までの値を順に要素に持つ1次元配列を生成
                np.random.shuffle(indices)   // 0～n_data-1までの値の要素順をシャッフルする
                x_shuffled = x[indices]      // xの要素順をindicesの要素の値順にシャッフルし、リストx_shuffledを生成
                y_shuffled = y[indices]      // yの要素順をindicesの要素の値順にシャッフルし、リストx_shuffledを生成

                # Pick up (batch_sizes) data randomly from original data.
                for i in range(0, n_data, batch_size): // iを0～n_dataまでの値でbatch_size間隔で生成する
                    x_batch = x_shuffled[i: i + batch_size] // x_shuffledのうち、batch_size分の要素をx_batchに代入し、リストにする
                    y_batch = y_shuffled[i: i + batch_size] // y_shuffledのうち、batch_size分の要素をy_batchに代入し、リストにする
                    yield x_batch, y_batch                  // この時点でのx_batch, y_batchをメソッド呼び出しもとにreturnする


            # The followings are same as tensorflow-basic12.py ----------------
            # Download the dataset of Boston house-prices
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.boston_housing.load_data()

            # global settings of matplotlib
            plt.rcParams['font.size'] = 10 * 3
            plt.rcParams['figure.figsize'] = [18, 12]
            plt.rcParams['font.family'] = ['IPAexGothic']

            # display histogram "house prices/$1000" vs "number of data"
            plt.hist(y_train, bins=20)        // bins:number of bar in histogram
            plt.xlabel('house prices/$1000')
            plt.ylabel('number of data')
            plt.show()

            # display plot "number of rooms" vs "house prices/$1000"
            plt.plot(x_train[:, 5], y_train, 'o')
                // 6th column is "number of rooms"
                // 'o' means circle marker
            plt.xlabel('number of rooms')
            plt.ylabel('house prices/$1000')
            plt.show()

            # preprocessing
            x_train_mean = x_train.mean(axis=0)
            x_train_std = x_train.std(axis=0)
            y_train_mean = y_train.mean()
            y_train_std = y_train.std()

            x_train = (x_train - x_train_mean) / x_train_std
            y_train = (y_train - y_train_mean) / y_train_std
            x_test = (x_test - x_train_mean) / x_train_std
            y_test = (y_test - y_train_mean) / y_train_std

            # display plot "number of rooms" vs "house prices/$1000" after preprocessing
            plt.plot(x_train[:, 5], y_train, 'o')
            plt.xlabel('number of rooms(after)')
            plt.ylabel('house prices/$1000(after)')
            plt.show()

            # define the inference model of Boston house-prices
            x = tf.placeholder(tf.float32, (None, 13), name='x')
            y = tf.placeholder(tf.float32, (None, 1), name='y')
            w = tf.Variable(tf.random_normal((13, 1)))
            pred = tf.matmul(x, w)

            # define the loss function and learning rate
            loss = tf.reduce_mean((y - pred) ** 2)
            optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=0.1
            )
            train_step = optimizer.minimize(loss)
            # The above are same as tensorflow-basic12.py ----------------

            # execute train step and repeatedly
            step = 0
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                for epoch in range(100):
                    for x_batch, y_batch in get_batches(x_train, y_train, 32):　// x_train, y_trainの要素をシャッフルし、32要素ずつ取り出す
                        train_loss, _ = sess.run(
                            [loss, train_step],
                            feed_dict={
                                x: x_batch,
                                y: y_batch.reshape((-1, 1))
                            }
                        )
                        print('step: {}, train_loss: {}'.format(step, train_loss))
                        step += 1

                pred_ = sess.run(
                    pred,
                    feed_dict={
                        x: x_test
                    }
                )
            ```
        * 上記のコードは最急降下法とほとんど同じであるが、sess.run()で処理するデータの単位が異なる。
        * ```get_batches()```では```x_train```, ```y_train```を渡し、各リストの要素をシャッフルして、ミニバッチに分割し、分割したミニバッチを呼ばれるたびに1つずつ返す処理をしている。
            * ```yield```を使うことにより、その時点での```x_batch```, ```y_batch```を返すことでその関数を抜けることができる。
        * ミニバッチを1つ分の処理単位を「イテレーション」、イテレーションを繰り返し(=```get_batches()```で全てのデータを取り出す)、データ全体を処理単位を「エポック」と呼ぶ。
        * するする以下はSGDによる学習の実装コードとなる。
* ニューラルネットワークとKeras
    * パーセプトロンとは
        * ニューラルネットワークの最も基本的なもので、複数の入力から1つの出力を行う。
        * 出力値は入力を重み付きで足し合わせた値が定められた閾値を超えている場合は1となり、超えていなければ、0となる。
        * パーセプトロンのようにある閾値を境に出力値が異なる関数をステップ関数と呼ぶ。
        * パーセプトロンは線形分離可能な問題を正しく表現することができる。
            * 線形分離可能とは直線で2種類の点の集まりに分けることができる状態のことであり、ANDやOR、NANDが該当する。
        * 一方、線形分離不可能な問題は表現できない。
            * XORは線形分離不可能なものの1つで、2つの入力のうち、一方が1のときに出力が1になり、それ以外の場合は出力が0になるものを指す。
            * ただし、XORは単一のパーセプトロンでは表現できないが、複数のパーセプトロンを組み合わせることで表現することができる。
        * 複数のパーセプトロンを組み合わせたものを多層パーセプトロン(MLP:Multi Layer Perceptron)と呼ぶ。
    * シグモイドニューロンとは
        * パーセプトロンはステップ関数(=0 or 1を出力する非連続関数)を使用しているが、関数が滑らかではないため、勾配法を使って学習することができない。
        * そこでステップ関数の代わりに滑らかな関数であるシグモイド関数を用いる。
        * シグモイド関数を用いたニューロンをシグモイドニューロンと呼ぶ。
    * 活性化関数とは
        * ステップ関数やシグモイド関数など重み付き入力の和(=ニューロンへの入力)をニューロンの出力に変換する関数のこと。
        * シグモイド関数は重み付き入力の和が大きい場合、および小さい場合はステップ関数とほとんど同じ値を取る。
        * また、ステップ関数とは異なり、滑らかに値が変化するため、勾配法を適用することが可能となるため、ニューラルネットワークの活性化関数として使われることがある。
    * 順伝播型ニューラルネットワークとは
        * ニューロンが層のように並び、隣接する層の間のみ結合するネットワークのことで、入力は順方向にのみ伝播して前の層に戻ることはないものを表す。
        * 最初と最後の層をそれぞれ入力層、出力層と呼び、その間の層を中間層などと呼ぶ。
        * 各ニューロンは複数の入力を受け取り、それらから重み付き入力の和を計算し、さらにその重み付き入力の和にバイアス項を加えて、活性化関数を使用して得られた値を出力する。
    * Kerasを使った順伝播型ニューラルネットワークの実装
        * 手書き文字認識のデータセットMNISTを使用する。
        * MNISTは28×28 pixelの0～9の数字が書かれた手書き文字70000枚のデータセットであり、各ピクセルは灰色の濃淡を表す0～255を取る(0は黒、255は白を表す)
        ```python
        // keras-feedforward_neural_network01.py
        from tensorflow.python.keras.datasets import mnist
        from tensorflow.python.keras.utils import to_categorical
        from tensorflow.python.keras.models import Sequential
        from tensorflow.python.keras.layers import Dense
        from tensorflow.python.keras.callbacks import TensorBoard

        # Download the dataset of Boston house-prices
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # Check the shapes of MNIST data to be downloaded.
        print('x_train.shape: ', x_train.shape)  // (60000, 28, 28)
        print('x_test.shape: ', x_test.shape)    // (10000, 28, 28)
        print('y_train.shape: ', y_train.shape)  // (60000, )
        print('y_test.shape: ', y_test.shape)    // (10000, )

        # Preprocessing - exchange the scale of data
        x_train = x_train.reshape(60000, 784)
        x_train = x_train/255
        x_test = x_test.reshape(10000, 784)
        x_test = x_test/255

        # Preprocessing - change class label to 1-hot vector
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)

        # Create the neural network
        # Input layer -> Hidden layer
        model = Sequential()
        model.add(
            Dense(
                units=64,
                input_shape=(784, ),
                activation='relu'
            )
        )

        # Hidden layer -> Output layer
        model.add(
            Dense(
                units=10,
                activation='softmax'
            )
        )

        # Learn the model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        tsb = TensorBoard(log_dir='./logs')
        history_adam=model.fit(
            x_train,
            y_train,
            batch_size=32,
            epochs=20,
            validation_split=0.2,
            callbacks=[tsb]
        )
        ```
        * まず、```tensorflow.python.keras.datasets```の```mnist```をインポートし、Boston house-pricesと同様、```load_data()```でMNISTのデータセットをダウンロードする。
        * ダウンロードした配列の形をshapeメソッドで確認する。
            * 学習データx_train/y_trainは60000枚あり、x_trainは28×28の画像データとなっている。
            * y_trainはx_trainに対する正解データが値として格納されている。
            * x_test/y_testは学習に使用せず、学習結果を評価するために使用する。データ自体はそれぞれx_train, y_trainと同じである。
        * 次に各データを前処理する。
        * x_train/x_testは28×28の画像を```reshape()```メソッドを使用して1次元にする。28×28=784なので、(60000, 28, 28)から(60000, 784)、(10000, 28, 28)から(10000, 784)への変換をそれぞれ行なう。
        * また、x_train/x_testはそれぞれ255で割り、各値を0～1に収まるfloat型に変換し、正規化する。
        * y_train/y_testは値として正解ラベルを持っているが、```to_categorical()```メソッドで正解ラベルをベクトルに変換する。ベクトルは正解ラベルに該当する要素を1にし、それ以外を0にして表現する。
        * このベクトル表現を1-hotベクトルと呼ぶ。
        * 次にKerasのSequential APIを使い、多層ニューラルネットワークを構築する。
            * Sequential APIはKerasでモデルを構築する方法の1つで、用意されているレイヤーを```add```メソッドで追加していくだけでモデルが構築できる。
        * 始めにSequentialクラスのインスタンス```model```を作り、```add```メソッドでDenceレイヤーを用い、全結合層を追加する。
        * Denceレイヤーでは引数として以下を指定し、入力層から中間層を構築する。
            * units=64　：　出力ニューロンの数(出力次元の大きさ)
            * input_shape=(784, )　：　入力のテンソルの形・大きさ
            * activation='relu'　：　活性化関数
        * 'relu'は活性化関数としてReLU関数を指定することを意味する。
            * ReLU関数はランプ関数と呼ばれ、入力が0より小さい場合、出力は0となり、入力が0以上の場合、出力は入力値となるような関数である。
            * シグモイド関数に比べて、ReLU関数を活性化関数関数として用いる方が、収束が速くなる場合があるため、よく用いられていることが多い。
        * 再度、```model```に```add```メソッドでDenceレイヤーを用いて、全結合層を追加する。
        * Denceレイヤーでは引数として以下を指定し、中間層から出力層を構築する。
            * units=10　：　出力ニューロンの数(出力次元の大きさ)
            * activation='softmax'　：　活性化関数
        * MNISTは0～9の10クラスのラベルが対象なので、出力層のニューロンの数として10を```units```で指定する。
        * ```input_shape```はKerasが自動で計算するため、上記のように省略可能。
        * 'softmax'は活性化関数としてsoftmax関数を指定することを意味する。
            * softmax関数はシグモイド関数を多出力に拡張させたもので、多クラス分類問題の活性化関数として用いられている。
            * softmax関数により、各出力を[0, 1]に収める正規化ができ、かつ各出力値の和を1にすることができる。
            * MNISTの認識モデルにおいて、出力層の各ニューロンの出力値が[0,1]に収まり、それらの出力値の合計が1になることは入力画像がどの数字であるかを認識させる作業において便利に使用することができる。
        * 次にモデルの学習を行なうための設定を以下の通り、```compile()```メソッドを使用して行なう。
            * optimizer='adam'　：　最適化アルゴリズムとしてadamを指定
            * loss='categorical_crossentropy'　：　損失関数として交差エントロピーを指定
            * metrics=['accuracy']　：　c
        * Boston house-pricesの学習において、最適化アルゴリズムはSGDを用いていたが、ここではAdamと呼ばれるアルゴリズムを使用する。Kerasでは```optimizer```に引数を指定するだけで最適化アルゴリズムを変更できる。
            * Adam(Adaptive Moment Estimation)は直近の勾配情報を利用する、といった工夫を実装しているアルゴリズムでSGDに比べ、収束が速いと言われている。
        * 損失関数としては、```'categorical_crossentropy'```を指定し、交差エントロピーを用いる。
            * 交差エントロピーは2つの確率分布間に定義される尺度で、分類問題の損失関数として用いられることが多い。
            * 分類問題においてはこの値が小さくなるように学習を行う。
        * 学習履歴に精度(accuracy)を追加するために```model```の```metrics```で```'accuracy'```を追加する。
            * 学習結果の履歴は```fit()```メソッドの返り値のインスタンスから取得できるが、デフォルトだとloss(訓練データセットのにおける損失の値)のみが履歴として取得できる状態になっている。
            * ```metrics```で```'accuracy'```を追加することにより、acc(訓練データにおけるモデルの精度)が履歴として追加できる状態となる。
            * また、後述の```fit()```メソッドにて検証データを設定すると、val_loss(検証データにおける損失)、val_acc(検証データに対するモデルの精度)も履歴として取得できるようになる。
        * 次にモデルの学習を以下の通り、```fit()```メソッドを使用して行なう。
            * x=x_train　：　学習用データ
            * y=y_train　：　学習用データのラベルデータ
            * batch_size=32　：　ミニバッチのサイズ
            * epochs=20　：　エポックサイズ
            * validation_split=0.2　：　学習用データのうち、検証用データとして使用する割合
            * callbacks=[tsb]　：　
        * ```fit()```メソッドの引数```x```, ```y```, ```batch_size```, ```epochs```にはそれぞれ、学習用データ(x_train)、学習用データのラベルデータ(y_train)、ミニバッチのサイズ、エポックサイズを指定する。
        * ```validation_split```には学習用データ(x_train)のうち、検証用データとして使用する割合を指定する。
            * 検証用データは学習したモデルが未知のデータに対してどの程度の予測性能を持っているかを測るために使用するデータのこと。
            * 0.2を指定することでx_trainのうちの80%をモデルの学習に使用し、残りの20%を学習したモデルの検証のために使用するようになる。
            * モデルの検証は1エポックが終了する毎に行われ、上述のval_loss, val_accなどを学習の履歴として残すことができる。
        * callbacksには```tsb = TensorBoard(log_dir='./logs')```を指定し、学習の結果をlogs以下のファイルに出力させる。学習後、ファイルを指定してTensorboardから学習結果を確認することができる。
        * 学習を行ない、Tensorboardから学習結果を確認する。
            * accは学習が進むにつれて増加し、lossは減少していることがわかる。これは学習が正しく進められていることを意味する。
            * ただし、この傾向は必ずしも未知のデータに対して高い予測性能を持っていることを意味するとは限らず、val_acc, val_lossの数値の傾向も合わせてみる必要がある。
            * val_acc, val_lossは学習では使用していない未知のデータ(検証データ)に対する精度、損失の値になるが、val_accが大きく、val_lossが小さくなっている方がよいと言える。
            * 結果を見ると、val_accが小さくなったり、val_lossが大きくなっているところもあり、これは「過学習」と呼ばれる。
            * 過学習は学習用データにモデルが過度に適合してしまうと起こり、未知のデータに対する予測性能が低下する現象である。
    * KerasのFunctional APIを使った順伝播型ニューラルネットワークの実装
        * 上述のSequential APIを使用した順伝播型ニューラルネットワークによるMNIST認識モデルをFunctional APIを使って実装する。
        * Sequential APIは便利だが、入力や出力が複数あるような複雑なモデルを記述することができないため、KerasではFunctional APIという別のインターフェースが用意されている。
        ```python
        // keras-feedforward_neural_network02.py
        from tensorflow.python.keras.datasets import mnist
        from tensorflow.python.keras.utils import to_categorical
        from tensorflow.python.keras.callbacks import TensorBoard
        from tensorflow.python.keras.models import Model
        from tensorflow.python.keras.layers import Input, Dense

        # The followings are same as keras-feedforward_neural_network01.py ----------------
        # Download the dataset of MNIST
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # Preprocessing - exchange the scale of data
        x_train = x_train.reshape(60000, 784)
        x_train = x_train/255
        x_test = x_test.reshape(10000, 784)
        x_test = x_test/255

        # Preprocessing - change class label to 1-hot vector
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)
        # The above are same as keras-feedforward_neural_network01.py ----------------

        # Create the neural network
        input = Input(shape=(784, ))
        middle = Dense(units=64, activation='relu')(input)
        output = Dense(units=10, activation='softmax')(middle)
        model = Model(inputs=input, outputs=output)

        # The followings are same as keras-feedforward_neural_network01.py ----------------
        # Learn the model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        tsb = TensorBoard(log_dir='./logs')
        history_adam = model.fit(
            x_train,
            y_train,
            batch_size=32,
            epochs=20,
            validation_split=0.2,
            callbacks=[tsb]
        )
        # The above are same as keras-feedforward_neural_network01.py ----------------
        ```
        * Sequential APIからFunctional APIへの変更に伴い、importするmoduleが変更となる。
            * Sequential→Modelへ変更
            * レイヤーはDenseに加え、Inputをimportするように変更
        * MNISTのデータセットの取得及び学習データ・テストデータ、及びそれらのラベルデータに対する前処理はSequential APIを使用した時と同じ。
        * ネットワークの構築に関して、Sequential APIではレイヤーを追加してモデルを構築していたのに対し、Functional APIでは同じレイヤーオブジェクトを追加していくが、オブジェクトを次の層の引数に与えることでモデルを構築する。
            * 上述の通り、まず```input```というテンソルを生成する。
            * 次に```input```を中間層のDenseの引数として渡し、```middle```というテンソルを生成する。
            * 最後に```middle```を出力層のDenseの引数として渡し、```output```というテンソルを生成する。
        * また、Functional APIでは、Modelクラスの引数```inputs```、```outputs```にそれぞれ入力のテンソル、出力のテンソルを指定することでモデルを構築することができる。
            * ```inputs```、```outputs```にはそれぞれ複数のテンソルを渡すことができ、入力や出力が複数あるネットワークの構築も容易に行うことができる。
        * ```compile()```、```fit()```メソッドによる学習のための設定及び学習はSequential APIで実装する場合と同じ。
* KerasによるCNNの実装
    * これまではKerasでMLPを実装してきた。
    * MNISTのデータセット(1つの画像が28×28のgrayscale)を使用していたため、入力層のニューロンは784個となっていた。
    * 中間層を64個のニューロンで構成したため、入力層のニューロンから中間層へのニューロンへの重みは1ニューロンあたり64個となり、バイアス値を含め、65個のパラメータを最適化していた。
    * これが、入力層のニューロンの数分パラメータが存在するため、ネットワーク全体では、65×784=50960個のパラメータを調整最適化することになる。
    * 











* 学習済みモデルの活用
* よく使うKerasの機能



## 参考
* 現場で使える！TensorFlow開発入門 Kerasによる深層学習モデル構築手法
https://www.shoeisha.co.jp/book/detail/9784798154121
