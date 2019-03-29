# 自然言語処理
* 日本語や英語など普段使っている言葉を自然言語(Natural Language)と言い、自然言語処理(NLP：Natural Language Processing)は自然言語を処理する分野、つまりコンピュータに人間の言葉を理解させるための技術のことである。
* 自然言語処理が扱う分野は多岐にわたるが、その本質はコンピュータに人間の言葉を理解させることにある。
* コンピュータが理解できる言語と言えば、プログラミング言語やマークアップ言語がある。
* それらの言語はコードの意味が一意に解釈できるように文法が定義されており、コンピュータはそのルールに従ってコードを解析する。
* 一般的なプログラミング言語は機械的で無機質であり、"固い言語"である。
* 一方、自然言語は同じ意味の文章でも様々な表現が可能であり、曖昧さがあったり、柔軟に意味や形が変わることから"柔らかい言語"であると言える。
* このような柔らかい言語である自然言語をコンピュータに理解させるのは難しい。
* 一方、その問題をクリアできれば、人にとって役立つことをコンピュータに行わせることができると言える。
* 実際にそのような例は多くあり、検索エンジンや機械翻訳、質問応答システム、文章の自動翻訳や感情分析などが挙げられる。
* 自然言語は文字によって構成され、また自然言語の言葉の意味は単語によって構成される。
* 単語は意味の最小単位であり、自然言語をコンピュータに理解させるためには、単語の意味を理解させることが重要であると言える。
* コンピュータに単語の意味を理解させるために、単語の意味をうまく捉えた以下の表現方法について考える。
    * シソーラスによる手法
    * カウントベースによる手法
    * 推論ベースの手法(word2vec)


# シソーラス
* 単語の意味を表す方法として、辞書のように1つ1つの単語に対してその意味を説明する方法が考えられる。
* このように単語の意味を定義することでコンピュータで単語の意味を理解する。
* ただし、これまでの自然言語処理の歴史では、人が使う辞書とは異なるシソーラス(thesaurus)と呼ばれるタイプの辞書が使われてきた。
* シソーラスは基本的には類語辞書であり、同じ意味の単語(同義語)、意味の似た単語(類義語)が同じグループに分類されている。
* 例えば、シソーラスを使うことでcarの同義語にはautomobileやmotocarなどが存在することがわかる。
* また、シソーラスでは、「上位と下位」、「全体と部分」などのより細かい関係性が単語の間で定義されている場合がある。
* 例えば、グラフ構造によって、各単語の関係性が定義されているケースがある。
* carという単語の上位概念としてmotor vehicleがあり、下位概念としてSUVやcompactがある。
* このように全ての単語に対して類義語の集合を作り、それぞれの単語の関係をグラフで表現することで、単語間の繋がりを定義することができる。
* この構造を使用することでコンピュータに単語間の関連性を教えることができる。


# 代表的なシソーラス - WordNet
* 自然言語処理分野におて、有名なシソーラスとして、WirdNetがある。
* WordNetはプリンストン大学で1985年に開発がスタートしており、これまで多くの研究に利用されてきた。
* WordNetを使うことで類義語を取得できたり、単語ネットワークを利用したりすることができる。
* さらに単語ネットワークを利用することで単語間の類似度を算出することも可能。


# シソーラスの問題点
* シソーラスにより、多くの単語に対して、同義語や階層構造の関係性を定義することによってコンピュータに単語の意味を教えることができる。
* 一方、それらを人の手でラベル付けすることには以下のような大きな欠点が存在する。
    * 時代変化に伴う対応が困難
        * 自然言語は生きており、絶えず新しい言葉が生まれ、古い言葉は忘れ去られる。
        * また時代変化に伴い、言葉の意味が変化する場合もある。
        * そのような単語の変化に対応するためにはシソーラスを人手で絶えず更新する必要がある。
    * 人手によるラベル付けのコストが高い
        * シソーラスを作るためには多くのコストがかかる。
        * 現存する英単語の総数は1000万語を超えると言われているが、それら全てに単語の関連づけをする必要がある。
    * 単語の細かいニュアンスを表現できない
        * シソーラスでは類義語として似た単語をグループ化するが、実際には似た単語であっても、それぞれニュアンスが少し異なる。
        * 単語ごとの微妙なニュアンスの差異はシソーラスでは表すことができない。
* 上記の通り、シソーラスを使う手法(単語の意味を人手によって定義する方法)には大きな問題がある。
* この問題を避けるため「カウントベースの手法」、ニューラルネットワークを使った「推論ベースの手法」がよく用いられる。
* これらの手法では、大量のテキストデータから自動的に単語の意味を抽出する。


# カウントベースの手法
* カウントベースの手法ではコーパス(corpus)を使用する。
* コーパスとは大量のテキストデータであるが、やみくもに集められたテキストデータではなく、自然言語処理の研究やアプリケーション開発のために目的をもって収集されたテキストデータのことを呼ぶ.
* コーパスはテキストデータに過ぎず、含まれる文章は人の手によって書かれたものである。
* これはコーパスに自然言語に対する人間の知識(文章の書き方、単語の選び方、単語の意味)が含まれていることを意味する。
* カウントベースの手法では人間の知識が詰まったコーパスから自動的かつ効率的にそのエッセンスを抽出することを行う。
* 自然言語処理分野で用いられるコーパスにはテキストデータに対してさらに追加の情報が付与されていることがある。
* 例えば、テキストデータの個々の単語に対して、品詞がラベル付けされているケースなどがある。
* その場合、コンピュータが扱いやすいようにコーパスが機構造などのデータ形式として構造化されていることが一般的である。
* 自然言語処理分野で用いられるコーパスには様々なものが存在し、WikipediaやGoogle Newsなどのテキストデータや、シェクスピアや夏目漱石などの偉大な作家の作品群がコーパスとして有名である。
* ここではまずはじめに1文からなる単純な以下のようなテキストをコーパスとして利用する。
    ```python
    >>> text = 'You say goodbye and I say hello.'
    ```
* 次にPythonの対話モードで小さなテキストデータに前処理(テキストデータを単語に分割し、その分割した単語を単語IDのリストへ変換)を行う。
    ```python
    >>> text = text.lower()
    >>> text = text.replace('.', '. ')
    >>> text
    'you say goodbye and i say hello .'

    >>> words = text,split(' ')
    >>> words
    ['you', 'say', 'goodbye', 'and', 'i', 'say', 'hello', '.']
    ```
* 上記では、まずlower()で全て小文字に変換し、ピリオドの前にスペースを挿入する。
    * 実際はより賢く汎用的なやり方として正規表現を使用する方法がある。
    * reモジュールをimportし、re.split('(\W+)?', text)とすることで単語単位に分割できる。
* 次にsplit()でスペースを区切り文字として、単語に分割し、wordsというリストに各単語を要素として保持する。
* これで元の文章を単語のリストとして使用できるようになる。
* ただ単語をテキストのまま操作するのは不便なため、単語にIDを振り、IDのリストとして使用できるようにする。
* まず以下の通り、単語のIDと単語の対応表をPythonのディクショナリで作成する。
    ```python
    >>> word_to_id = {}
    >>> id_to_word = {}
    >>>
    >>> for word in words:
            if word not in word_to_id:
                new_id = len(word_to_id)
                word_to_id[word] = new_id
                id_to_word[new_id] = word
    ```
* id_to_wordにはキーを単語ID、値を単語として保持し、word_to_idにはキーを単語、値を単語IDとして保持する。
* if文では単語がword_to_idにまだ保持されていない場合にword_to_idの要素数をnew_idとして払い出し、word_to_idにそのnew_idを値として追加する。
* id_to_wordにはそのnew_idをキーとしてwordを追加する。
* 上述のtextの文はid_to_word、word_to_idにそれぞれ以下のように保持される。
    ```python
    >>> id_to_word
    {0:'you', 1:'say', 2:'goodbye', 3:'and', 4:'i', 5:'hello', 6:'.'}
    >>> word_to_id
    {'you':0, 'say':1, 'goodbye':2, 'and':3, 'i':4, 'hello':5, '.':6}
    ```
* これらのディクショナリを使うことにより、単語から単語IDを検索したり、単語IDから単語を検索したりできる。
    ```python
    >>> id_to_word[1]
    'say'
    >>> word_to_id['hello ']
    5
    ```
* 最後に以下の通り、単語のリストwordsを単語IDのリストcorpusに変換し、その結果をNumpy配列にして保持する。
    ```python
    >>> import numpy as np
    >>> corpus = [word_to_id[w] for w in words]
    >>> corpus = np.array(corpus)
    >>> corpus
    array([0,1, 2, 3, 4, 1, 5, 6])
    ```
* for文の内包表記ではwordsの先頭から要素の値を順に取り出してwに格納し、そのwをキーとしてword_to_idの値、つまり単語IDを取得する。
* 取得した単語idはcorpusにリストとして保持される。
* 次の処理でcorpusはnp.array()でNumpy配列に変換される。
* ここまでの処理を前処理として、preprocess()という関数にしてまとめて実装すると、以下のようになる。
    ```python
    def preprocess(text):
        text = text.lower()
        text = text.replace('.', '. ')
        words = text,split(' ')

        word_to_id = {}
        id_to_word = {}
        for word in words:
            if word not in word_to_id:
                new_id = len(word_to_id)
                word_to_id[word] = new_id
                id_to_word[new_id] = word

        corpus = np.array([word_to_id[w] for w in words])

        return corpus, word_to_id, id_to_word
    ```
* この関数を使用すると、コーパスの前処理は以下のようになる。
    ```python
    >>> text = 'You say goodbye and I say hello.'
    >>> corpus, word_to_id, id_to_word = preprocess(text)
    ```
* ここまででコーパスを扱う準備が整ったので、これを用い、単語の意味を抽出する。
* ここでカウントベースの手法を用い、単語をベクトルで表すことを行なう。
* ベクトル表現の例として「色」を考える。
* 色には様々な固有の名前が付けられている一方、RGBの3成分がその色にどれだけ含まれるかによって色を表現することもできる。
* 前者は色の数だけ異なる名前を命名することになるが、後者は色を3次元ベクトルで表すことになる。
* ここで、RGBによる色のベクトル表現の方が正確に色を指定できると言える。
* また例えば、(R, G, B) = (201, 23, 30)であれば、その色の名前を知るよりも、赤系の色だとイメージがしやすいと言える。
* 加えて、色同士の関連性・類似性もベクトル表現の方が容易に判断でき、また定量化も容易にできる。
* このような色のベクトル表現を単語でも行えないかを考える。
* つまり、コンパクトかつ理にかなったベクトル表現を「単語」というフィールドにおいて構築することを考える。
* これが構築できれば、「単語の意味」を的確に捉えたベクトル表現が可能となる。
* このベクトル表現を単語の分散表現と呼ぶ。
* 単語の分散表現では単語を固定長のベクトルで表現し、そのベクトルは密なベクトル(ベクトルの各要素が0でない実数で構成されるベクトル)で表現される。
* これまでの自然言語処理の研究で行われてきた単語をベクトルで表現する手法は「単語の意味は周囲の言語によって形成される」というアイデアに基づいているものがほとんどである。
* これは分布仮説(distributional hypothesis)と呼ばれ、単語自体に意味はなく、その単語のコンテキスト(文脈)によって、単語の意味が形成される、というものである。
* 意味的に同じ単語は同じような文脈で多く出現する。
    * 例)"I drink beer."、"We drink wine."などの文章におけるdrink
* また、以下のような文章があれば、"guzzle"が"drink"と同じような文脈で使われること、意味が近い単語である、ということもわかる。
    * "I guzzle beer."、"We guzzle wine."
* 注目する単語に対するその周囲に存在する単語をコンテキストと呼び、コンテキストに周囲のどこまでの単語を含めるかのサイズをウィンドウサイズ(Window size)と呼ぶ。
* つまりウィンドウサイズが2の場合、左右のそれぞれ2単語がコンテキストとなる。
* ウィンドウサイズは左右均等である必要はなく、どちらか一方だけをコンテキストとする、といったことや分の区切りを考慮したコンテキストを考える、といったこともできる。
* 上記をもとに分布仮説に基づいて単語をベクトルで表現する方法を考える。
* もっとも単純な方法はある単語の周囲にどのような単語がどの程度出現するかをカウントし、それを集計することである。
* まず前述のpreprocess()関数を使って下準備を行う。
    ```python
    import sys
    sys.path.append('..')
    import numpy as np
    from common.util import preprocess

    text = 'You say goodbye and I say hello.'
    corpus, word_to_id, id_to_word = preprocess(text)

    print(corpus)
    # [0,1, 2, 3, 4, 1, 5, 6]

    print(id_to_word)
    # {0:'you', 1:'say', 2:'goodbye', 3:'and', 4:'i', 5:'hello', 6:'.'}
    ```
* 上記より、textは単語数は8、語彙数は7であることがわかる。
* この各単語について、ウィンドウサイズを1とし、コンテキストに含まれる単語の頻度を数えると以下のようになる。
    |         | you   | say   | goodbye | and   | i     | hello | .     |
    |:-------:|:-----:|:-----:|:-------:|:-----:|:-----:|:-----:|:-----:|
    | you     | 0     | 1     | 0       | 0     | 0     | 0     | 0     |
    | say     | 1     | 0     | 1       | 0     | 1     | 1     | 0     |
    | goodbye | 0     | 1     | 0       | 1     | 0     | 0     | 0     |
    | and     | 0     | 0     | 1       | 0     | 1     | 0     | 0     |
    | i       | 0     | 1     | 0       | 1     | 0     | 0     | 0     |
    | hello   | 0     | 1     | 0       | 0     | 0     | 0     | 1     |
    | .       | 0     | 0     | 0       | 0     | 0     | 1     | 0     |
* 1行目の'you'は隣り合っている'say'のみがコンテキストとなるため、1となり、それ以外の単語は0となる。
* これは同時に'you'という単語が[0, 1, 0, 0, 0, 0, 0]というベクトルで表現できることを意味している。
* これを全ての語彙に対して行ない作成されるテーブルを共起行列(co-occurence matrix)と呼ぶ。
    * 共起とは、ある単語がある文中に出現したとき、その文中に別の単語が頻繁に出現すること。
* この共起行列をnp.arrayで定義すると以下のようになる。
    ```python
    C = np.array([
        [0, 1, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 1, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 0]
    ], dtype=np.int32)
    ```
* この共起行列を使うことで以下の通り、各単語のベクトルを取得することができる。
    ```
    print(C[0]) # 単語IDが0のベクトル
    # [0, 1, 0, 0, 0, 0, 0]

    print(C[4])  # 単語IDが4のベクトル
    # [0, 1, 0, 1, 0, 0, 0]

    print(C[word_to_id['goodbye']]) # 'goodbyeのベクトル'
    # [0, 1, 0, 1, 0, 0, 0]
    ```
* これを踏まえ、コーパスから共起行列を作る関数create_co_matrix()を実装する。
    * create_co_matrix(corpus, vocab_size, window_size=1)
        * corpus：単語IDのリスト
        * vocab_size：語彙数
        * window_size：ウィンドウサイズ
    ```python
    def create_co_matrix(corpus, vocab_size, window_size=1):
        corpus_size = len(corpus)
        co_matrix = np.zeors((vocab_size, vocab_size), dtype=np.int32)

        for idx, word_id in enumerate(corpus):
            for i in range(1, window_size+1):
                left_idx = idx - i
                right_idx = idx + i

                if left_idx >= 0:
                    left_word_id = corpus[left_idx]
                    co_matrix[word_id, left_word_id] += 1
                
                if right_idx < corpus_size:
                    right_word_id = corpus[right_idx]
                    co_matrix[word_id, right_word_id] += 1

        return co_matrix
    ```
* まず語彙数×語彙数の大きさの2次元配列を全要素を0で初期化そて生成する。
* 次にコーパス中の各単語に対して、そのウィンドウに含まれる単語をカウントしていく。
* ただし、コーパスの右端・左端のはみ出しチェックを行い、はみ出していない単語のみカウントする。
* この関数を用いることで任意の大きさのコーパスに対する共起行列を生成することができる。
* 次にベクトル間の類似度を計測する方法を考える。
* 一般的には様々な方法があり、ベクトルの内積やユークリッド距離などが代表的な方法として挙げられる。
* 単語のベクトル表現の類似度に関しては、コサイン類似度(cosine similarity)がよく用いられる。
* コサイン類似度はベクトルx, yに対して、以下の(2.1)で定義される。
    * similarity(x, y) = (x・y) / (||x| ||y||) = (x1y1 + x2y2 +・・・xnyn) / (√x1^2+・・・+xn^2 √y1^2+・・・+yn^2)　(2.1)
        * x = (x1, x2, ... , xn)
        * y = (y1, y2, ... , yn)
* (2.1)は分子にベクトルの内積、分母に各ベクトルのノルムがある。
* ノルムはベクトルの大きさを表したのもので、ここではL2ノルム(ベクトルの各要素の2乗和の平方根)を用いている。
* これはベクトルを正規化して内積を取っていると言える。
* コサイン類似度は直感的には2つのベクトルがどれだけ同じ方向を向いているかを表すものである。
* 2つのベクトルが完全に同じ方向を向いている場合、コサイン類似度は1となり、完全に逆向きだと-1となる。
* これにより、コサイン類似度を実装すると以下のようになる。
    ```python
    def cos_similarity(x, y):
        nx = x / np.sqrt(np.sum(x**2))　＃xの正規化
        ny = y / np.sqrt(np.sum(y**2))　＃yの正規化

        return np.dot(nx, ny)
    ```
* 引数x, yはNumPy配列とし、x, yをそれぞれ正規化した後、両者の内積を求めている。
* 一方、上記の実装だとx, ｙに0ベクトルが渡されてくると、0除算が発生してしまう。
* ここでは、小さな値epsを1e-8とし、引数で指定できるようする。
* 指定されたepsはnx, nyの分母の計算に必ず加算される値として実装する。
* これを踏まえたcos_similarity()の改良版は以下の通りとなる。
    ```python
    def cos_similarity(x, y, eps=1e-8):
        nx = x / np.sqrt(np.sum(x**2)) + eps)　＃xの正規化
        ny = y / np.sqrt(np.sum(y**2)) + eps)　＃yの正規化

        return np.dot(nx, ny)
    ```
* このcos_similarity()改良版を用いて、"you"と"i"の単語ベクトルの類似度を求める実装は以下の通りになる。
    ```python
    import sys
    sys.path.append('..')
    from common.util import preprocess, create_co_matrix, cos_similarity

    text = 'You say goodbye and I say hello.'
    corpus, word_to_id, id_to_word = preprocess(text)
    vocab_size = len(word_to_id)
    C = create_co_matrix(corpus, vocab_size)

    c0 = C[wprd_to_id['you']] # 'you'の単語ベクトル
    c1 = C[wprd_to_id['i']]   # 'i'の単語ベクトル
    print(cos_similarity(c0, c1))
    # 0.7071067691154799
    ```
* "you"と"i"の単語ベクトルの類似度は0.70...と言える。
* コサイン類似度は1から-1までの値を取るので、この値は比較的高い値(=類似正がある)と言える。
* 次にある単語がクエリとして与えられたとき、そのクエリに対して類似した単語を上位から順に表示する関数most_similar()を考える。
    * most_similar()
        * query：クエリ(単語)
        * word_to_id：単語から単語IDへのディクショナリ
        * id_to_word：単語IDから単語へのディクショナリ
        * word_matrix：単語ベクトルをまとめた行列で各行に対応する単語のベクトルが格納(共起行列)
        * top：取得する類似度のランキング
* 実装は以下の通り。
    ```python
    def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
        # Pick up the query
        if query not in word_to_id:
            print('%s is not found' % query)
            return

        print('\n[query] ' + query)
        query_id = word_to_id[query]
        query_vec = word_matrix[quiery_id]

        # calculate cosine similarity
        vocab_size = len(id_to_word)
        similarity = np.zeros(vocab_size)
        for i in range(vocab_size):
            similarity[i] = cos_similarity(word_matrix[i], query_vec)
        
        # Output the value of cosine similarity in descending order 
        count = 0
        for i in (-1 * similarity).argsort():
            if id_to_word[i] == query:
                continue
            print('%s: %s' % (id_to_word[i], similarity[i]))

            count +=1
            if count >= top:
                return
    ```
* まずはじめに渡されてきたqueryがword_to_idのキーに存在するかをチェックする
* 存在する場合は処理をそのqueryを表示し、処理を続行する。
* word_to_idからqueryをキーにその単語のIDを取得する。取得した単語IDを使い、word_matrixからその単語の単語ベクトルを取得し、query_vecに保持する。
* 次にコサイン類似度similarityを計算する。まず、id_to_wordの大きさ分のベクトルを要素0で初期化する。
* cos_similarity()にword_matrixの各行とquery_vecを渡し、コサイン類似度を計算し、各行の値をベクトルsimilarityの要素として保持する。
* 最後にコサイン類似度をその値が高い順に表示するためにargsort()関数を使用する。
* argsort()はNumpy配列の要素を小さい順にソートし、ソート後のインデックスを返す関数である。
    ```python
    >>> x = np.array([100, -20, 2])
    >>> x.argsort()
    # array([1, 2, 0])
    ```
* よって、similarityの各要素に-1をかけ、argsort()関数を適用すると、その返り値は値の大きい要素順のインデックスが変えることになる。
* これによって、for文を回し、id_to_word, similarityを表示する。
* このfor文をtopで渡された回数分回すことで、
* コサイン類似度が高い順にその値と単語を表示することができる。
* 次にこのmost_similar()関数を使用する例は以下の通り。例ではクエリとして'you'を指定する。
    ```python
    import sys
    sys.path.append('..')
    from common.util import preprocess, create_co_matrix, most_similar

    text = 'You say goodbye and I say hello.'
    corpus, word_to_id, id_to_word = preprocess(text)
    vocab_size = len(word_to_id)
    C = create_co_matrix(corpus, vocab_size)

    most_similar('you', word_to_id, id_to_word, C, top=5)
    ```
* これを実行すると以下のような結果が得られる。
* この結果から'you'に近い単語は3つあり、'goodbye', 'i', 'hello'であることがわかる。
    ```
    [query] you
    goodbye: 0.7071067691154799
    i: 0.7071067691154799
    hello: 0.7071067691154799
    say: 0.0
    and: 0.0
    ```
* 'i'は'you'と同じ人称代名詞であるため、コサイン類似度が高いことはよく理解できる。
* 一方、'goodbye'や'hello'のコサイン類似度が高いには説明がつかないが、これはコーパスのサイズが極端に短いことが原因と考えられる。


# カウントベースの手法の改善
* 前述の通り、単語の共起行列を作り、単語をベクトルで表現することができたが、さらに共起行列を改善する。
* 前述の共起行列の要素は2つの単語が共起した回数を表している。
* ただ、これは特に高頻度単語に目を向けると、あまり良い性質を持たないことがわかる。
* 例えば、コーパスにおいて、「the」、「car」の共起を考える。
* その場合、コーパスでは「...the car...」というフレーズは多くみられることが考えられ、共起回数は大きくなる。
* 一方、「car」と「drive」は単語間に明らかに強い関連性があるにもかかわらず、出現回数だけを見ると、「car」は「drive」よりも「the」の方が強い関連性を持ってしまうことになる。
* このような問題を解決するため、相互情報量(PMI：Pointwise Mutual Information)と呼ばれる指標が使われる。
* PMIは確率変数x, yに対して次の式で定義され、値が大きいほどx, yの関連性が高いと言える。
    * PMI(x, y) = log_2 ((P(x, y)) / (P(x)P(y)))　　(2.2)
        * P(x)：xが起きる確率(単語xがコーパスに出現する確率)
        * P(y)：yが起きる確率(単語yがコーパスに出現する確率)
        * P(x, y)：xとyが同時に起きる確率(単語x, yが共起する確率)
* 例えば、10000個の単語からなるコーパスで「the」が100回出現するとする。P('the') = 100/10000 = 0.01となる。
* 次にy = 'car'とし、「the」と「car」が10回共起したとすると、P('the', 'car') = 10/10000 = 0.001となる。
* 共起行列(共起した単語の回数を要素に持つ行列)をC、単語x, yの出現回数及び単語x,yの共起回数をそれぞれC(x), C(y), C(x,y)とし、コーパスに含まれる単語数をNとすると、(2.2)は以下のように表すことができる。
    * PMI(x, y) = log_2 ((P(x, y)) / (P(x)P(y))) = log_2 ((C(x,y)/N) / (C(x)/N・C(y)/N)) = log_2 ((C(x,y)・N) / (C(x)C(y))　　(2.3)
* (2.3)が導出できたことにより、PMIは共起行列から求めることが可能であることがわかる。
* この式を用い、具体的なコーパスに関してPMIを算出する。
* N=10000のコーパスにおいて、「the」「car」「drive」がそれぞれ1000回、20回、10回出現したとする。
* また、「the」と「car」、「car」と「drive」の共起はそれぞれ10回、5回とする。
* この時点で従来の共起行列の観点では、「the」と「car」の方が「car」と「drive」より関連性が高いことが言える。
* 一方、「the」と「car」、「car」と「drive」のPMIを算出すると、以下のようになる。
    * PMI('the', 'car') = log_2 (10・10000)/(1000・20) ≒ 2.32
    * PMI('car', 'drive') = log_2 (5・10000)/(20・10) ≒ 7.97
* 上記より、PMIの観点では、「car」と「drive」の方が「the」と「car」より関連性が高いという結果になる
* これは上述の通り、PMIにより単語単独の出現回数が考慮されてことによると言える。
* 一方、PMIは2つの単語の共起回数が0になる場合、log_2 0 = ∞となってしまう。
* よって、実際には以下で示す、正の相互情報量(PPMI：Positive PMI)が使われる。
    * PPMI(x, y) = max(0, PMI(x,y))　　(2.6)
* これにより、PMIの値がマイナスの場合は、それを0として扱うことができ、単語間の関連度を0以上の実数で表すことができる。
* 共起行列をPPMI行列(要素がPPMI値の行列)に変換する関数を実装すると、以下の通りとなる。
    ```python
    def ppmi(C, verbose=False, eps=1e-8):
        M = np.zeros_like(C, dtype=np.floar32)
        N = np.sum(C)
        S = np.sum(C, axis=0)
        total = C.shape[0] * C.shape[1]
        cnt = 0

        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                pmi = np.log2(C(i, j) * N / (S[j]*S[i]) + eps)
                M[i, j] = max(0, pmi)

                if verbose:
                    cnt+=1
                    if cnt % (total//100) == 0:
                        print('%lf%% done', %(100*cnt/total))
        
        return M
    ```
* この関数を用いて、共起行列をPPMI行列に変換する実装は以下のようになる。
    ```python
    import sys
    sys.path.append('..')
    import numpy as np
    from common.util import preprocess, create_co_matrix, cos_similarity, ppmi

    text = 'You say goodbye and I say hello.'
    corpus, word_to_id, id_to_word = preprocess(text)
    vocab_size = len(word_to_id)
    C = create_co_matrix(corpus, vocab_size)
    W = ppmi(C)

    np.set_printoptions(precision=3)
    print('covariance matrix')
    print(C)
    print('-'*50)
    print('PPMI')
    print(W)
    ```
* 上記を実行すると以下の結果が得られる。
    ```
    covarience matrix
    [[0 1 0 0 0 0 0]
     [1 0 1 0 1 1 0]
     [0 1 0 1 0 0 0]
     [0 0 1 0 1 0 0]
     [0 1 0 1 0 0 0]
     [0 1 0 0 0 0 1]
     [0 0 0 0 0 1 0]]
     --------------------------------------------------
     PPMI
    [[0.     1.807  0.     0.     0.     0.     0.   ]
     [1.807  0.     0.807  0.     0.807  0.807  0.   ]
     [0.     0.807  0.     1.807  0.     0.     0.   ]
     [0.     0.     1.807  0.     1.807  0.     0.   ]
     [0.     0.807  0.     1.807  0.     0.     0.   ]
     [0.     0.807  0.     0.     0.     0.     2.807]
     [0.     0.     0.     0.     0.     2.807  0.   ]]
    ```
* これにより、PPMI行列を取得することができ、共起行列より良い指標の単語ベクトルを取得できたと言える。
* ただ、PPMI行列はこーますの語彙数が増えるにつれ、各単語の単語ベクトルの次元数が増えていくという問題がある。
* つまり、コーパスに含まれる語彙数が10万個の場合、そのベクトルの次元数も同様に10万となることを意味する。
* PPMI行列はその要素の多くが0であり、これはほとんどの要素が重要ではなく、各要素の持つ重要度が低いことを意味する。
* このように0要素が多いベクトルはノイズに弱く、頑健性に乏しいという欠点がある。
* これらの問題に対してはベクトルの次元削減(dimensionality reduction)を行なう。
* ただし、単純に次元を削減するのではなく、重要な情報をできるだけ残したうえで削減する必要がある。
* ここでの次元削減ではデータの分布を見て、重要な軸を見つけることを行なう。
* 例えば、2次元の座標で表されているデータ点を1次元の座標で表すとする。
* その際はデータの広がりを考慮し、その広がりが大きな軸を見つけ、新たな軸を導入する。
* 新しい軸を見つけ、導入する場合、各データ点の値は新しい軸へ射影された値に変換される。
* データの広がりを考慮した軸が取れていれば、1次元の値に変換されても、データの本質的な差異は失わずに捉えたままにできる。
* ベクトル中のほとんどの要素が0である行列は「疎な行列」と呼ばれる。
* 疎な行列から重要な軸を見つけ、より少ない次元で表現し直すことで、次元が少なくほとんどの要素が0でない行列(密な行列)に変換することを考える。
* 次元削減にはいくつかの方法があり、ここでは特異値分解(SVD：Singular Value Decomposition)を用いる。
* SVDでは任意の行列Xを3つの行列U, S, Vの積へと分解する。式で書くと以下の通りとなる。
    * X = U・S・V^T　　(2.7)
* ここで、U及びVは直交行列(=両行列の積が単位行列)、Sは対角行列(対角成分以外の成分が全て0である行列)となる。
* 直交行列Uは単語空間の軸を形成しており、対角行列Sの対角成分には「特異値」が大きい順に並んでいる。
* この特異値は単語空間の軸の重要度とみなすことができる。
* つまり、行列Sの特異値が小さいものは重要度が低いとみなし、それらの要素を削ることで次元を削減し、元の行列の近似した行列を取得する。
* これは単語のPPMI行列から重要度の低い要素を削減した単語ベクトルU'を取得することを意味する。
* SVDの実装はNumpyのlinalgモジュールにあるsvdメソッドで実行できる。
    * linalgはlinear algebra(線形代数)を表す。
* SVDを適用する実装は以下の通り。
    ```python
    import sys
    sys.path.append('..')
    import numpy as np
    from common.util import preprocess, create_co_matrix, ppmi

    text = 'You say goodbye and I say hello.'
    corpus, word_to_id, id_to_word = preprocess(text)
    vocab_size = len(word_to_id)
    C = create_co_matrix(corpus, vocab_size, window_size=1)
    W = ppmi(C)

    U, S, V = np.linalg.svd(W)
    ```
* 上記にて得られる行列UにSVDによって変換された密な単語ベクトル表現が格納されている。
* 共起行列、PPMI行列、SVDで得られる行列を表示すると以下の通り。
    ```python
    print(C[0])　　# 共起行列
    # [0 1 0 0 0 0 0]

    print(W[0])　　# PPMI行列
    # [0.     1.807  0.     0.     0.     0.     0.   ]

    print(U[0])　　# SVD
    # [3.409e-01  -1.110e-16  -1.205e-01  -4.441e-16  0.000e+00  -9.323e-01  2.226e-16]
    ```
* 上記より、疎なベクトルW[0]がSVDによって密なベクトルU[0]に変換されていることがわかる。
* この密なベクトルを次元削減する場合、その先頭の要素を必要な分だけ取り出すことで実現できる。
* 例えば、2次元ベクトルに削減する場合は以下のようになる。
    ```python
    print(U[0, :2])
    # [3.409e-01  -1.110e-16]
    ```
* 次に各単語を2次元ベクトルで表すことにし、それを以下の実装でグラフにプロットしてみる。
    ```python
    for word_id in word_to_id.items():
        plt.annotate(word, (U[word_id, 0], U[word_id, 1]))
    
    plt.scatter(U[:,0], U[:,1], alpha=0.5)
    plt.show()
    ```
* plt.annotate(word, x, y)により、2次元グラフ上の座標(x,y)にwordというテキストを描画する。
* 結果を見ると、「goodbye」と「hello」、「you」と「i」が近い位置にいることがわかる。
* SVDの計算は行列のサイズをNとして場合、O(N^3)というオーダーの計算量になる。
* これは現実的に非常に大きな計算量となるため、通常はTruncated SVDのような高速化手法を用いるのが一般的である。
* Truncated SVDでは特異値が小さなものを切り捨てることで高速化を図っている。
* 上記の結果は、コーパスが小さく結果が正しく出ていない部分もある。
* 以下では、PTBデータセットというより大きなコーパスを使って同じ処理を行って結果を確認する。
* PTBはPenn Treebankと呼ばれるコーパスのことで、提案手法の品質を測定するためのベンチマークとしてよく利用されている。
* PTBコーパスはWord2vecの発明者であるTomas MikolovのWebページで用意されているものでテキストファイルでて供されているが、以下のような前処理が施されている。
    * レアな単語を\<unk>に置換
    * 具体的な数字をNに置換
* 以下ではこれら前処理が施されているテキストデータをPTBコーパスとして使用する。
* PTBコーパスは1つの文が1行ごとに保存されており、ここでは各文を連結したものを1つの大きな時系列データとして扱うことにする。
* 加えて各文の終わりに\<eos>という特殊文字(eos=end of sentence)を挿入する。
* Penn Treebankのデータセットを扱う実装は以下の通り。
    ```python
    import sys
    sys.path.append('..')
    from dataset import ptb

    corpus, word_to_id, id_to_word = ptb.load_data('train')

    print('corpus size:', len(corpus))
    print('corpus[:30]', corpus[:30])
    print()
    print('id_to_word[0]:', id_to_word[0])
    print('id_to_word[1]:', id_to_word[1])
    print('id_to_word[2]:', id_to_word[2])
    print()
    print('word_to_id['car']:', word_to_id['car'])
    print('word_to_id['happy']:', word_to_id['happy'])
    print('word_to_id['lexus']:', word_to_id['lexus'])
    ```
* 上記を実行すると、以下の結果が得られる。
    ```
    corpus size:929589
    corpus[:30]: [0 1 2 3 4 5 6 7 8...27 28 29]

    id_to_word[0]:aer
    id_to_word[1]:banknote
    id_to_word[2]:berlitz

    word_to_id['car']: 3856
    word_to_id['happy']: 4428
    word_to_id['lexus']: 7426
    ```
* corpusには単語IDのリストが格納され、id_to_word, word_to_idにはそれぞれ、単語IDから単語への変換ディクショナリ、単語から単語IDへの変換ディクショナリが格納される。
* ptb.load_data()ではPTBのデータをloadする。引数で'train', 'test', 'valid'を渡すことにより、それぞれ学習用、テスト用、検証用のデータが指定できる。
* PTBデータセットに対してカウントベースの手法を適用することを考える。
* ここではより高速なSVDを使用するため、sklearnをimportし、randomized_svd()メソッドを使用する。
* randomized_svd()では乱数を使ったTruncated SVDで特異値の大きなものだけに限定して計算し、通常のSVDよりも高速に計算を行うことが可能。
* 実装は以下の通り。
    ```python
    import sys
    sys.path.append('..')
    import numpy as np
    from common.util import most_similar, create_co_matrix, ppmi
    from dataset import ptb

    window_size = 2
    wordvec_size = 100

    corpus, word_to_id, id_to_word = ptb.load_data('train')
    vocab_size = len(word_to_id)
    print('counting co-occurrence...')
    C = create_co_matrix(corpus, vocab_size, window_size=1)
    print('calculating PPMI...')
    W = ppmi(C, verbose=True)

    print('calculating SVD...')
    try:
        # truncated SVD
        from sklearn.utils.extmath import randomized_svd
        U, S, V = randomized_svd(W, n_components=wordvec_size, n_iter=5, random_state=None)
    except ImportError:
        # SVD
        U, S, V = np.linalg.svd(W)
    
    word_vecs = U[:, :wordvec_size]

    querys = ['you', 'year', 'car', 'toyota']
    for query in querys:
        most_similar(query, word_to_id, id_to_word, word_vecs, top=5)
    ```
* 上記を実行すると、以下の結果が得られる。
    ```
    [query] you
     i: 0.702039909619
     we: 0.699448543998
     've: 0.554828709147
     do: 0.534370693098
     else: 0.512044146526

    [query] year
     month: 0.731561990308
     quarter: 0.658233992457
     last: 0.622425716735
     earlier: 0.607752074689
     next: 0.601592506413
    
    [query] car
     luxury: 0.620933665528
     auto: 0.615559874277
     cars: 0.569818364381
     vehicle: 0.498166879744
     corsica: 0.472616831915
    
    [query] toyota
     motor: 0.738666107068
     nissan: 0.677577542584
     motors: 0.647163210589
     honda: 0.628862370943
     lexus: 0.604740429865
    ```
* 上記の結果を見てみると、まず「you」というクエリに対して、「i」、「we」といった人称代名詞が上位を占めており、文法的に共通の単語が該当している。
* 「year」がクエリの場合は「month」や「quarter」、「car」がクエリの場合は「auto」や「vehicle」などの類義語が得られていることがわかる。
* このように単語の意味的な点と文法的な点において、似た単語同士が近いベクトルとして表されていることがわかる。


# word2vec
* ここまではカウントベースの手法によって、単語の分散表現を得てきたが、それに代わる強力な手法として推論ベースの手法がある。
* 推論ベースの手法ではニューラルネットワークを用いており、有名な手法としてword2vecがある。
* 以降ではword2vecの仕組みを理解し、実装することを行なう。
* ただまずは、シンプルなword2vecを実装することで処理効率よりもわかりやすさを重視した実装を行っていく。
* これまでは単語の分散表現(ベクトル化)の手法として、大きく「カウントベースの手法」と「推論ベースの手法」が研究されてきた。
* 両者のアプローチは大きくことなるが、どちらも分布仮説(単語自体に意味はなく、その単語のコンテキストによって、単語の意味が形成されるという仮説)に基づいている。
* まずここまで用いてきたカウントベースの手法では周囲の単語の頻度によって単語を表現してきた。
* 具体的には共起行列を作り、さらにSVDを適用することで密なベクトルを得てきた。
* ただ、このカウントベースの手法には大きな問題があり、その問題は規模が大きなコーパスを扱う際に顕在化する。
* 現実世界でコーパスを扱う場合、語彙数は非常に多くなり得る。
* 例えば、英語の場合、語彙数は100万を優に超えると言われている。
* この場合、カウントベースの手法では、100万×100万の巨大な行列を作ることになり、その巨大な行列に対して、SVDを行なうことは現実的ではないと言える。
* SVDはn×nの行列に対して、O(n^3)の計算コストがかかる。これはnの大きさの3乗に比例して計算時間が増えていくことを意味する。
* このような計算はスーパーコンピューターでも困難であるため、実際は近似的な手法や疎な行列の性質を利用して処理速度の向上させるのが一般的である。
* ただそのような高速化処理を行っても、多くの計算李s-すと計算時間が必要となる。
* カウントベースの手法はコーパス全体の統計データ(共起行列やPPMIなど)を利用して1回の処理(SVDなど)で単語の分散表現を得ている。
* 一方、推論ベースの手法では、ニューラルネットワークを用いる場合、ミニバッチで学習するのが一般的である。
* ミニバッチで学習するとは、1度に少量(ミニバッチ)の学習サンプルを使って、重みを繰り返し更新していくことを意味する。
* つまり、カウントベースの手法が学習データを1度にまとめて処理するのに対し、推論ベースの手法では、学習データの一部を使って逐次的に学習していると言える。
* これは語彙数が多いコーパスにおいて、SVDなどの計算量が膨大で難しい処理を行う場合でも、ニューラルネットワークではデータを小分けにして学習を行うことが可能なことを意味している。
* さらにニューラルネットワークの学習では、複数マシン・複数GPUの利用による並列計算も可能であり、全体の学習も高速化することができる。
* これらの点において、推論ベースの手法の方が優れていると言える。
* 推論ベースの手法では、以下のように周囲の単語が与えられた特に「？」にどんな単語が出現するかを推論する作業と言える。
    * you ? goodbye and I say hello.
* また、このような推論問題を解き、学習することが「推論ベースの手法」が扱う問題である。
* このような推論問題を繰り返し解くことで単語の出力パターンを学習する。
* つまり推論ベースの手法では何らかのモデルが登場し、そのモデルにニューラルネットワークを用いる。
* このモデルはコンテキスト情報を入力として受け取り、出現しうる各単語の出現する確率を出力する。
* モデルによって正しい推測ができるようにコーパスを用い、モデルの学習を行う。
* その学習の成果として単語の分散表現が得られる、というのが推論ベースの手法の全体図と言える。
* ニューラルネットワークを使って単語を処理する場合、"you"や"say"のような単語をそのまま処理することはできず、固定長のベクトルに変換する必要がある。
* 固定長べのベクトルへの変換手法の1つとして、one-hot表現がある。
* one-hot表現とは、ベクトル中の要素のうち、1つだけが1で残りは0であるベクトル表現のことを指す。
* "You say goodbye and I say hello."という1分をコーパスとして扱う場合を考える。
* このコーパスには語彙が7個存在し、この各単語のone-hot表現は以下のようになる。
    | 単語    | ID  | one-hot表現            |
    |:-------:|:---:|:---------------------:|
    | you     | 0   | (1, 0, 0, 0, 0, 0, 0) |
    | say     | 1   | (0, 1, 0, 0, 0, 0, 0) |
    | goodbye | 2   | (0, 0, 1, 0, 0, 0, 0) |
    | and     | 3   | (0, 0, 0, 1, 0, 0, 0) |
    | i       | 4   | (0, 0, 0, 0, 1, 0, 0) |
    | hello   | 5   | (0, 0, 0, 0, 0, 1, 0) |
    | .       | 6   | (0, 0, 0, 0, 0, 0, 1) |
* 上記の通り、単語のone-hot表現はコーパスの語彙数分の要素数を持つベクトルを用意し、単語IDに該当する要素を1、残りの要素を0として生成している。
* このように単語を固定長のベクトルに変換することで、ニューラルネットワークの入力層はニューロンの数を固定することができる。
* 上記のコーパスを学習データの対象とした場合、入力層は7つのニューロンで構成する。
* つまり、各ニューロンがそれぞれ7つの単語に対応する。
* これにより、単語をベクトルで表現することでその単語(ベクトル)はニューラルネットワークを構成する様々なレイヤで処理することができる。
* ここでone-hot表現で表された単語に対して全結合層で変換する場合を考える。
* 全てのノードには中間層とのつながりがありそこには重みが存在するため、中間層のニューロンは入力層のニューロンとの重み付き和となる。
* また、ここではニューロンのバイアスは省略する。
* バイアスを用いない全結合層は「行列の積」の計算に相当し、多くのディープラーニングのフレームワークでは全結合層の生成時にバイアスを用いない選択をすることが可能となっている。
* ここまでの全結合層による変換は以下のように実装される。　※中間層は3ノードとする
    ```python
    import numpy as np

    C = np.array([[1, 0, 0, 0, 0, 0, 0]])  # 入力
    W = np.random.randn(7, 3)              # 重み
    h = np.dot(c, W)                       # 中間ノード
    print(h)
    # [[0.70012195  0.25204755  0.79774592]]
    ```
* 上記では単語IDが0の単語をone-hot表現で表し、それを入力として、全結合層によって変換をしている。
* 全結合層の計算はバイアスを省略しているため、行列の積で行うことができ、Numpyのnp.dot()を用いている。
* また、cとWの積の部分はcがone-hot表現(特定の要素のみが1でそれ以外は0)のため、重みWの行ベクトル(=1行n列行列)を抜き出す作業に相当する。
    * 単語IDが0の場合は、one-hot表現されたベクトルcは0番目の要素のみが1である。
    * このベクトルと任意の値を要素として持っている重みWの積を計算した場合、cの0番目の要素との積を求める部分だけが0ではなく、Wの要素の値そのものになる。
    * ベクトルcの0番目の要素との積はWの0行目の要素と行われる。他の要素との積は行っても、0となるため、結果として、Wの0行目の要素を抜き出すことになる。
* 行ベクトルを抜き出すために行列計算を行うのは非効率だが、その改良に関しては後述する。
* また、この実装をMatMulレイヤを使用すると以下のようになる。
    ```python
    import sys
    sys.path.append('..')
    import numpy as np
    from common.layers import MatMul

    C = np.array([[1, 0, 0, 0, 0, 0, 0]])
    W = np.random.randn(7, 3)
    layer = MatMul(W)
    h = layer.forward(c)
    print(h)
    # [[0.70012195  0.25204755  0.79774592]]
    ```
* MatMulレイヤに重みWを設定し、forward()メソッドで順伝播の処理を行うことで、積を計算する。
* 以下ではシンプルなword2vecを実装する。つまり、「モデル」としてニューラルネットワークを組み込む。
* ここでは、word2vecで提案されているcontinuous bag-of-words(CBOW)というモデルを使用する。
    * word2vecという用語は本来、プログラムやツール類を指して用いられていたが、文脈によってはニューラルネットワークのモデルを指す場合もある。
    * 厳密にはCBOWモデルとskip-gramモデルという2つのモデルがword2vecで使用されるニューラルネットワークである。
* CBOWモデルはターゲット(中央の単語)をコンテキスト(周囲の単語)を推測することを目的としたニューラルネットワークである。
* CBOWモデルを正確な推測ができるように訓練することで単語の分散表現を獲得することができる。
* CBOWモデルへの入力はコンテキストであり、['you', 'goodbye']のような単語のリストで表されるが、これをone-hot表現に変換し、CBOWモデルが処理できるようにしてから用いる。
* CBOWモデルのネットワークは入力層を2つ(7ノード×2)とし、中間層(7ノード×1)を経て、出力層(7ノード×1)で構成される。
* 2つの入力層から中間層への変換は同じ全結合層によって行い、その時の重みはW_in(7×3行列)とする。
* また、中間層から出力層への変換は別の全結合層によって行い、その時の重みはW_out(3×7行列)とする。
* コンテキストとして2つの単語を考える場合、上記のように入力層が2つとなる。つまりN個のコンテキストを扱う場合、N個の入力層を用意する必要がある。
* 中間層の各ニューロンでは各入力層の全結合による変換後の値が平均されたものになる。
* つまり、入力層が2つの場合、それぞれの変換後の値がh1, h2とすると、中間層のニューロンは(h1+h2)/2となる。
* 出力層には7個のニューロンがあり、それらのニューロンは各単語に対応している。
* 出力層のニューロンは各単語の「スコア」であり、その値が高いほど、その単語の出現確率が高いと言える。
* ただし、ここでのスコアは確率として解釈される前の値となっており、実際はこのスコアにSoftmax関数を適用して、確率として値を取得する。
* 入力層から中間層への変換は全結合層によって行われるが、この全結合層の重みW_inは7×3の行列でこの重みが単語の分散表現となる。
* 重みW_inの各行には各単語の分散表現が格納されていると考え、学習を重ねることでコンテキストから出現する単語をうまく推測できるように各単語の分散表現が更新されていく。
* また、このようにして得られたベクトルは単語の意味もうまくエンコードされている。
* ここで中間層のニューロン数は入力層のニューロン数よりも少なくする必要がある。
* これは中間層には単語を予測するために必要な情報をコンパクトに収める必要があり、またその結果として密なベクトル表現が得られるためである。
* CBOWモデルの推論処理(スコアを算出する処理)の実装は以下の通り。
    ```python
    import sys
    sys.path.append(..)
    import numpy as np
    from common.layers import MatMul

    # Context data sample
    c0 = np.array([[1, 0, 0, 0, 0, 0, 0]])
    c1 = np.array([[0, 0, 1, 0, 0, 0, 0]])

    # Initialize wegiht
    W_in = np.random.randn(7, 3)
    W_out = np.random.randn(3, 7)

    # Create layer
    in_layer0 = MatMul(W_in)
    in_layer1 = MatMul(W_in)
    out_layer = MatMul(W_out)

    # Forward propagation
    h0 = in_layer0.forward(c0)
    h1 = in_layer0.forward(c1)
    h = 0.5 * (h0+h1)
    s = out_layer.forward(h)

    print(s)
    # [[ 0.30916255  0.45060817  -0.77308656  0.22054131  0.15037278
    #   -0.93659277 -0.59612048]]
    ```
* まず重みW_in、W_outをランダム値で初期化し、入力側のMatMulレイヤをコンテキストの数だけ(ここでは2つ)生成し、また出力層側のMatMulレイヤを1つ生成する。
* ここで、入力側のMatMulレイヤは重みW_inを共有して生成する。
* る偽にMatMulレイヤのforward()メソッドで中間データを生成し、出力層側のMatMulレイヤのforward()メソッドによって、各単語のスコアを算出する。
* 上記の通り、CBOWモデルは活性化関数を用いないシンプルな構成になっている。
* 次にCBOWモデルの学習を行う。
* まず、CBOWモデルは出力層において各単語のスコアを出力するが、そのスコアに対してSoftmax関数を適用することで、その言語の確率を得ることができる。
* この確率は前後の単語であるコンテキストが与えられた時にその中央にどの単語が出現するかを表すものとなる。
* 学習データとして、例えばコンテキストとして「you」、「goodbye」が与えられ、正解ラベルが「say」として与えられる。
* このとき、十分に学習された重みをもつネットワークがあれば、確率を表すニューロンにおいて、「say」に対応するニューロンの確率が高くなると言える。
* よって、CBOWモデルの学習は正しい予測ができるように重みを調整することを意味する。
* それにより、W_in、W_outに単語の出現パターンを捉えたベクトルが学習される。
* CBOWモデルの学習で得られる単語の分散表現は単語の意味的な点、文法的な点において、人間の直感と合致するケースが多いことがこれまでの研究から分かっている。
* CBOWモデルの学習ではコーパスにおける単語の出現パターンを学んでいるため、コーパスが異なれば、学習で得られる単語の分散表現も異なるものになる。
* 例えば、コーパスとして、スポーツのニュース記事だけを使う場合と音楽のニュース記事だけを使う場合では得られる単語の分散表現は大きく異なるものとなる。
* 上記の推論処理の実装に実装を追加して学習を行うためには、多クラス分類を行うためにSoftmax関数と交差エントロピー誤差を導入する必要がある。
* 上述の実装で得られるスコアをSoftmax関数で確率に変換し、その確率と正解データから交差エントロピー誤差を求め、それを損失として学習を行う。
* よって、CBOWモデルに対して、SoftmaxレイヤとCross Entropy Errorレイヤを追加することで学習が可能となるが、ここでは、以前に実装済みのSoftmax with Lossというレイヤを使用して実装する。
* word2vecで使用されるネットワークには上述の通り、入力側の全結合層の重みW_inと出力側の全結合層の重みW_outがある。
* W_inは各行が各単語の分散表現となっており、W_outは単語の意味がエンコードされたベクトルが格納されていると考えることができる。
* さらに出力側の重みW_outは列方向(縦方向の並び)に各単語の分散表現が格納されている。
* このように入力側・出力側の両方の重みに含まれる単語の分散表現はどちらを利用するべきか？を考える。
* 入力側だけ使用する、出力側だけ使用する、両方使用する、などが考えられる。
* また、両方使用する場合は、どのように2つを組み合わせるかでさらにいくつかの手法が考えられる。
* word2vecに関しては、入力側の重みだけを使用するのが最も用いられている方法である。
* いくつかの研究でW_inのみを使用することの有効性が示されている。
* 学習で用いる学習データはこれまでと同様、「You say goodbye and I say hello.」をコーパスとして使用する。
* word2vecで用いるニューラルネットワークの入力はコンテキストであり、正解データはコンテキストで囲まれた中央の単語(ターゲット)である。
* つまり、学習の目標はニューラルネットワークにコンテキストを入力したときにターゲットの単語が出現する確率を高くすることである。
* 上記を踏まえ、コーパスからコンテキストとターゲットを作ることを考える。
* 以下では両端の単語を除くコーパス中の全ての単語をターゲットとし、そのコンテキストを抜き出している。
    | contexts     | target  |
    |:-------------|:--------|
    | you, goodbye | say     |
    | say, and     | goodbye |
    | goodbye, I   | and     |
    | and, say     | I       |
    | I, hello     | say     |
    | say, .       | hello   |
* 上記のcontextsの各行がニューラルネットワークの入力となり、targetの各行が正解データとなる。
* まず学習を行う準備として、コーパスからコンテキストとターゲットを作成する関数を作成する。
* はじめにpreprocess()を用い、以下のようにコーパスのテキストを単語IDに変換する。
    ```python
    import sys
    sys.path.append('..')
    from common.util import preprocess

    test = 'You say goodbye and I say hello.'
    corpus, word_to_id, id_to_word = preprocess(text)
    print(corpus)
    # [0  1  2  3  4  5  6]

    print(id_to_word)
    # {0:'you', 1:'say', 2:'goodbye', 3:'and', 4:'i', 5:'hello', 6:'.'}
    ```
* 次にこの単語IDの配列から上記で行ったのと同様、contextsとtargetを抜き出すことを単語IDベースで行い、それぞれを単語IDのリストとして格納する。
    | contexts  | target  |
    |:----------|:--------|
    | [[0  2]   | [1      |
    |  [1  3]   |  2      |
    |  [2  4]   |  3      |
    |  [3  1]   |  4      |
    |  [4  5]   |  1      |
    |  [1  6]]  |  5]     |
* これにより、コンテキストは6×2の2次元配列(次元はコンテキストの大きさ)、ターゲットは6個の要素を持つ1次元の配列となる。
* 以上を踏まえ、コーパスからコンテキストとターゲットを作成する関数create_contexts_target()を実装すると以下のようになる。
    ```python
    def create_contexts_target(corpus, window_size=1):
        target = corpus[window_size:-window_size]　#corpus[1:-1]
        contexts = []

        for idx in range(window_size, len(corpus)-window_size):
            cs = []
            for t in range(-window_size, window_size+1):　# -1, 0, 1
                if t == 0:
                    continue
                cs.append(corpus[idx+t])
            contexts.append(cs)
        
        return np.array(contexts), np.array(target)
    ```
* まず、targetをcorpusの両端の単語以外をリストとして持つため、corpus[window_size:-window_size]として格納する。
* 次にfor文を入れ子で回す。外のfor文でcorpusを両端の単語を除く先頭から回し、内のfor文でtargetとその前後のコンテキストを回し、target以外の要素を取得する処理を行う。
* 関数としては、返り値をcontextsとtargetをそれぞれNumpyの多次元配列として返している。
* この関数を使う実装は以下のようになる。
    ```python
    contexts, target = create_contexts_target(corpus, window_size=1)

    print(contexts)
    # [[0  2]
    #  [1  3]
    #  [2  4]
    #  [3  1]
    #  [4  5]
    #  [1  6]]

    print(target)
    # [1  2  3  4  1  5]
    ```
* 上記により、単語IDのコーパスからコンテキストとターゲットを生成することができた。
* 次にこれらをCBOWモデルに渡すために単語IDで構成されているコンテキストとターゲットをone-hot表現に変換する。
    | contexts            | target            |
    |:--------------------|:------------------|
    | [[[1 0 0 0 0 0 0]   | [[0 1 0 0 0 0 0]  |  
    |   [0 0 1 0 0 0 0]]  |                   |
    |  [[0 1 0 0 0 0 0]   |  [0 0 1 0 0 0 0]  |
    |   [0 0 0 1 0 0 0]]  |                   |
    |  [[0 0 1 0 0 0 0]   |  [0 0 0 1 0 0 0]  |
    |   [0 0 0 0 1 0 0]   |                   |
    |  [[0 0 0 1 0 0 0]   |  [0 0 0 0 1 0 0]  |
    |   [0 1 0 0 0 0 0]]  |                   |
    |  [[0 0 0 0 1 0 0]   |  [0 1 0 0 0 0 0]  |
    |   [0 0 0 0 0 1 0]]  |                   |
    |  [[0 1 0 0 0 0 0]   |  [0 0 0 0 0 1 0]] |
    |   [0 0 0 0 0 0 1]]] |                   |
* これにより、コンテキストは6×2×7の3次元配列、ターゲットは6×7の2次元配列となる。
* このone-hot表現への変換のconvert_one_hot()関数を用いて以下のような実装となる。
    ```python
    import sys
    sys.path.append('..')
    from common.util import preprocess, create_contexts_target, convert_one_hot

    text = 'You say goodbye and I say hello.'
    corpus, word_to_id, id_to_word = preprocess(text)

    contexts, target = create_contexts_target(corpus, window_size=1)

    vocab_size = len(word_to_id)
    target = convert_one_hot(target, vocab_size)
    contexts = convert_one_hot(contexts, vocab_size)
    ```
* ここまででCBOWモデルの学習のための準備が完了し、以下では学習に関する実装を行う。
* 実装するクラスはSimpleCBOWとしての実装は以下の通りとなる。
    ```python
    import sys
    sys.path.append('..')
    import numpy as np
    from common.layers import MatMul, SoftmaxWithLoss

    class SimpleCBOW:
        def __init__(self, vocab_size, hidden_size):
            V, H = vocab_size, hidden_size

            # Initialize weight
            W_in = 0.01 * np.random.randn(V, H).astype('f')
            W_out = 0.01 * np.random.randn(H, V).astype('f')

            # Create layers
            self.in_layer0 = MatMul(W_in)
            self.in_layer1 = MatMul(W_in)
            self.out_layer = MatMul(W_out)
            self.loss_layer = SoftmaxWithLoss()

            # Create lists of all weight and gradient
            layers = [self.in_layer0, self.in_layer1, self.out_layer]
            self.params, self.grads = [], []
            for layer in layers:
                self.params += layer.params
                self.grads += layer.grads
            
            # Set the distributed expression of words to member variables.
            self.word_vecs = W_in
    ```
* まずコンストラクタは上記のような実装となる。
* 引数として語彙数vocab_sizeと中間層のニューロンの数hidden_sizeを受け取る。
* 重みの初期値として、W_in、W_outを生成し、NumPy配列のデータ型astype('f')とし、要素を32ビットの浮動小数点数とする。
* 次に入力側のMatMulレイヤを2つ、出力側のMatMulレイヤを1つ、Softmax with Lossレイヤを1つそれぞれ作成する。
* 入力側のMatMulレイヤ数はコンテキストで使用する単語数と同じ分だけ生成する(ここでは2つ)
* 次にニューラルネットワークで使われるパラメータと勾配をメンバー変数params, gradsにリストとして格納する。
* また、W_in(単語の分散表現)もメンバーword_vecsに格納する。
* 上記の実装だと同じ重みを複数のレイヤで共有していることになるため、メンバー変数paramsのリストには、同じ重みが複数存在することになる。
* これにより、AdamやMomentumなどのoptimize処理が正しく動作しなくなる可能性がある。
* よって、Trainerクラス内部でパラメータの更新時にパラメータの重複を取り除く処理を行っている。
* 次にニューラルネットワークの順伝播処理を行うメソッドforward()を実装すると以下の通りとなる。
    ```python
        def forward(self, contexts, target):
            h0 = self.in_layer0.forward(contexts[:, 0])
            h1 = self.in_layer1.forward(contexts[:, 1])
            h = (h0 + h1) * 0.5
            score = self.out_layer.forward(h)
            loss = self.loss_layer.forward(score, target)
            return loss
    ```
* forward()メソッドは引数として、contextsとtargetの2つを取り、損失(loss)を返す。
* contextsはここでは3次元のNumPy配列となり、(6,2,7)=(コンテキストペアの数,コンテキストサイズ,語彙数)となる。
* また、targetはここでは2次元NumPy配列となり、(6,7)=(コンテキストペアの数,語彙数)となる。
* 次にニューラルネットワークの逆伝播処理を行うメソッドbackward()を実装すると以下の通りとなる。
    ```python
        def backward(self, dout=1):
            ds = self.loss_layer.backward(dout)
            da = self.out_layer.backward(ds)
            da *= 0.5
            self.in_layer1.backward(da)
            self.in_layer0.backward(da)
            return None
    ```
* 各パラメータの勾配をメンバー変数gradsにまとめ、順伝播・逆伝播の実装を行なったため、forward()メソッドを呼び、次にbackward()メソッドを呼ぶことでgradsに保持されている勾配が更新することができる。
* 以下では、SimpleCBOWクラスの学習の実装を行う。
* CBOWモデルの学習も通常のニューラルネットワークの学習と全く同じで、まず学習データを準備してニューラルネットワークに与え、勾配を算出し、重みを逐一アップデートしていく処理を行う。
    ```python
    import sys
    sys.path.append('..')
    from common.trainer import Trainer
    from common.optimizer import Adam
    from simple_cbow import SimpleCBOW
    from common.util import preprocess, create_contexts_target, convert_one_hot

    window_size = 1
    hidden_size = 5
    batch_size = 3
    max_epoch = 1000

    text = 'You say goodbye and I say hello.'
    corpus, word_to_id, id_to_word = preprocess(text)

    vocab_size = len(word_to_id)
    contexts, target = create_contexts_target(corpus, window_size=1)
    target = convert_one_hot(target, vocab_size)
    contexts = convert_one_hot(contexts, vocab_size)

    model = SimpleCBOW(vocal_size, hidden_size)
    optimizer = Adam()
    trainer = Trainer(model, optimizer)

    trainer.fit(contexts, target, max_epoch, batch_size)
    trainer.plot()
    ```
* パラメータを更新するoptimizerには様々な手法があるが、ここではAdamを使用する。
* Trainerクラスではニューラルネットワークの学習を行なう。
* 具体的には学習データからミニバッチを生成し、それをニューラルネットワークに与えて勾配を算出、その勾配をoptimizerに渡してパラメータを更新する処理を行う。
* このような一連の処理をTrainerクラスに集約することで学習のためのコードをシンプルにすることができる。
* 最後のplot()メソッドで実行結果をグラフで確認する。
* 縦軸を損失、横軸をiterationとしてグラフを表示すると、学習を重ねるごとに損失が減少していくことがわかる。
* ここで、学習が終わった後の重みパラメータを確認する。
* 入力側のMatMulレイヤの重みを以下のコードで取り出す。
    ```python
    word_vecs = model.word_vecs
    for word_id, word in id_to_word.items():
        print(word, word_vecs[word_id])
    ```
* 入力側のMatMulレイヤの重みはSimpleCBOWのメンバー変数word_vecsに格納されているので、これを取得し、for文で単語とその単語のベクトル表現を取得する。
* 実行結果は以下のようになる。
    ```
    you [-0.9031807  -1.0374491  -1.4682057  -1.3216232  0.93127245]
    say [ 1.2172916   1.2620505  -0.07845993  0.07709391 -1.2389531]
    goodbye [-1.0834033  -0.8826921  -0.33428606  -0.5720131  1.0488235]
    and [ 1.0244362  1.0160093  -1.6284224  -1.6400533  -1.0564581]
    i [ -1.0642933  -0.9162385  -0.31357735 -0.5730831  1.041875]
    hello [-0.9018145  -1.035476  -1.4629668  -1.3058501  0.9280102]
    . [ 1.0985303  1.1642815  1.4365371  1.3974973  -1.0714306]
    ```
* これにより、コーパスに含まれる単語の密なベクトル表現を得ることができたと言える。
* ただ、これらは'You say goodbye and I say hello.'という小さなコーパスから得られた結果に過ぎない。
* より大きく実用的なコーパスを使用することで、よりよい単語の分散表現を得ることができる。
* その場合、これまでのやり方だと処理速度の点で問題が発生する。これまでの手法には処理効率の点でいくつかの問題があるためである。
* 以降ではこれまでのシンプルなCBOWモデルに対して改良を加え、実用的なCBOWモデルを実装していく。
* まず、CBOWモデルを「確率」という視点から見てみる。
* 「確率」はP(・)と表され、Aという事象が起こる確率はP(A)と表記される。
* また、「AとBが同時に起こる確率」つまり同時確率はP(A,B)と表記し、「Bが起こった後にAが起こる確率」つまり事後確率はP(A|B)と表記される。事後確率は「Bという情報が与えられたときにAが起こる確率」と解釈することもできる。
* これらを踏まえ、CBOWモデルを確率の表記で記述することを考える。
* CBOWモデルはコンテキストを与えると、ターゲットとなる単語の確率を出力するという処理を行うので、ここでは、w1, w2,...wTという単語列のコーパスとウィンドウサイズが1のコンテキストを考える。
* ここで、コーパス中のt番目の単語をwtをすると、コーパスは以下のように表される。
    ```
    w1  w2  w3 ...  wt-1  wt  wt+1  ...  wT-1  wT
    ```
* このとき、「コンテキストとしてwt-1とwt+1が与えられたとき(=B)」に「ターゲットがwtになる(=A)」確率は事後確率を用いて以下のようになる。
    * P(wt | wt-1,wt+1)　　(3.1)
* これはつまり、CBOWは(3.1)をモデル化していることを意味する。
* また、(3.1)を用いることで、CBOWモデルの損失関数も簡潔に表すことができる。
* まず、交差エントロピー誤差は以下のように表すことができる。
    * L = - Σ tk・logyk　　(1.7)
        * yk：k番目に対応する事象が起こる確率
        * tk：教師ラベル(one-hotベクトル)
* このエントロピー誤差に「ターゲットがwtになる確率」をあてはめると、tkのt番目の要素のみが1、それ以外の要素は0となり、yt = P(wt | wt-1,wt+1)とすることができる。
* これを考慮すると、(3.1)、(1.7)から以下のようになる。
    * L = -logP(wt | wt-1,wt+1)　　(3.2)
* 上記より、CBOWモデルの損失関数は(3.1)の確率に対してlogを取り、その値にマイナスをつけたものであると言える。これは負の対数尤度(negative log likelihood)と呼ぶ。
* (3.2)は1つのサンプルデータに関する損失関数となるが、これをコーパス全体に拡張すると損失関数は以下のようになる。
    * L = -(1/T)・ΣlogP(wt | wt-1,wt+1)　　※t = 1, 2, ..., T　　(3.3)
* これを踏まえると、CBOWモデルの学習とはこの(3.3)で表される損失関数を小さくすることであり、重みパラメータが単語の分散表現となる。
* word2vecでは、CBOWモデルの他にskip-gramと呼ばれるモデルがある。
* skip-gramはCBOWで扱うコンテキストとターゲットを逆転させたモデルである。
* CBOWモデルとskip-gramモデルで解くべき問題を比較すると以下のようになる。
    * CBOWモデル：　you  ?  goodbye and I say hello.
    * skip-gramモデル：　?  say  ?  and I say hello.
* CBOWモデルはコンテキストが複数あり、それらから中央のターゲットの単語を推測するのに対し、skip-gramモデルは中央のターゲットの単語から周囲の複数の単語(コンテキスト)を推測する。
* よって、skip-gramモデルのレイヤー構成はCBOWモデルのレイヤー構成の逆の構成になる。
* つまり、skip-gramモデルでは入力層が1つで出力層がコンテキストの数だけ存在するような構成となる。
* 出力層では個別に損失を求め、それらを合計したものを最終的な損失とする。
* 次にskip-gramモデルを確率の表記を用いて表すことを考える。
* ターゲットとなる単語をwt、ターゲットの単語のコンテキストとなる単語をwt-1、wt+1とすると、skip-gramモデルは以下のように表される。
    * P(wt-1,wt+1 | wt)　　(3.4)
* これは「ターゲットwtが与えられたときにwt-1、wt+1が同時に起こる確率」を意味する。
* ここでは、wt-1、wt+1の間に関連性がないと仮定して、以下のように分解した表現として考える。
    * P(wt-1,wt+1 | wt) = P(wt-1 | wt) P(wt+1 | wt)　　(3.5)
* さらにCBOWモデルと同様、(1.7)の交差エントロピー誤差の式を適用し、skip-gramモデルの損失関数を導くと以下のようになる。
    * L = -logP(wt-1,wt+1 | wt) = -logP(wt-1 | wt) P(wt+1 | wt) = -(logP(wt-1 | wt) + logP(wt+1 | wt))　　(3.6)
    * ※ここでは、logxy = logx + logyの性質を用いて変換している。
* 上式の通り、skip-gramモデルの損失関数はコンテキスト分の損失をそれぞれ求め、合計したものであると言える。
* CBOWモデルと同様、損失関数をコーパス全体に拡張すると、以下のようになる。
    * L = -(1/T)Σ(logP(wt-1 | wt) + logP(wt+1 | wt))　　※t = 1, 2, ..., T　　(3.7)
* (3.3)、(3.7)を比較することで、CBOWモデルとskip-gramモデルの違いがわかる。
* skip-gramモデルはコンテキストの数だけ推測するため、損失関数は各コンテキストで求めた損失の総和を求めているのに対し、CBOWモデルは1つのターゲットの損失を求めている。
* これまでのCBOWモデルとskip-gramモデルでは、skip-gramモデルを使用する方がよい。
* skip-gramモデルの方が、多くの場合、単語の分散表現の精度がよいこと、またコーパスが大規模になると、低頻出単語や類推問題の性能において、優れた結果が得られる傾向があることが知られているからである。
* 一方、学習速度の点ではCBOWモデルの方が、skip-gramモデルよりも高速である。
* これはskip-gramモデルはコンテキストの数だけ損失を求めるため、計算コストが大きくなるためである。
* skip-gramモデルは1つの単語からその周囲の単語を予測することになるが、直感的にこの問題は難しいと言える。
* CBOWモデルではコンテキストから1つの単語を予測するので、直感的には比較的簡単な問題と言える。
* つまり、skip-gramモデルの方がより難しい問題に取り組み、学習しているため、より優れた単語の分散表現を得られると言える。
* 次のカウントベースの手法と推論ベースの手法(特にword2vec)の手法を比較する。
* カウントベースの手法はコーパス全体の統計データから1回の学習で単語の分散表現を得ていたが、推論ベースの手法では、コーパスの一部を何度も見ながらミニバッチとして学習し、単語の分散表現を得ている。
* まず、語彙に新しい単語を追加するケースで単語の分散表現の更新作業が発生する場合を考える。
* この場合、カウントベースの手法では、ゼロから計算をし直す必要がある。
* 単語の分散表現を少し修正したい場合であっても、再度、共起行列を作り直し、SVDを行なう一連の作業が必要となる。
* 一方、推論ベースの手法であるword2vecでは、パラメータの再学習を行うことができる。
* つまり、これまで学習した重みを初期値として再学習することでこれまでの学習経験を失うことなく、単語の分散表現の更新を効率的に行うことができる。
* これらを踏まえると、語彙が追加された場合は推論ベースの手法の方がよいと言える。
* 次にカウントベースの手法と推論ベースの手法それぞれで得られる単語の分散表現の性質や精度を比較する。
* 分散表現の性質に関しては、カウントベースの手法では主に単語の類似性がエンコードされることがわかっている。
* word2vecでは単語の類似性に加え、さらに複雑な単語間のパターンも捉えることができることがわかっている。
    * word2vecで「king - man + woman = queen」のような類推問題を解けることが知られている。
* ただ単語の類似性に関する定量評価ではカウントベースの手法と推論ベースの手法に関しては、優劣をつけられないことが判明している。
* これはハイパーパラメータの依存度が大きいためである。
* また、推論ベースの手法とカウントベースの手法には関連性があることがわかっている。
* 具体的にはskip-gramモデル、negative samplingを用いたモデルは、カウントベースの手法で生成するコーパス全体の共起行列に対して、特殊な行列分解をしているのと同じであることが示されている。
* また、word2vec以降、推論ベースとカウントベースの手法を融合させたGloVeという手法も提案されている。
* この手法では、コーパス全体の統計データの情報を損失関数に取り入れてミニバッチ学習を行っている。


# word2vecの高速化
* ★★～P.130★★


# Method
* P.64：text.lower()
* P.64：text.replace('.', '. ')
* P.64：text,split(' ')
* P.66：[word_to_id[w] for w in words]　※内包表記
* P.69：sys.path.append('..')
* P.72：enumerate(corpus)
* P.75：if query not in word_to_id:
* P.75：x.argsort()
* P.80：np.set_printoptions(precision=3)
* P.85：word_to_id.items()
* P.85：plot.annotate(word, (U[word_id, 0], U[word_id, 1]))
* P.85：plt.scatter(U[:,0], U[:,1], alpha=0.5)
* P.99：np.random.randn(7, 3)
* P.99：np.dot(c, W)
* P.113：range(start, stop)　※start～stop-1までの整数を生成





