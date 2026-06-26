# pylint: disable=import-outside-toplevel
# pylint: disable=line-too-long
# flake8: noqa
"""
Escriba el codigo que ejecute la accion solicitada en cada pregunta.
"""


def pregunta_01():
    """
    La información requerida para este laboratio esta almacenada en el
    archivo "files/input.zip" ubicado en la carpeta raíz.
    Descomprima este archivo.

    Como resultado se creara la carpeta "input" en la raiz del
    repositorio, la cual contiene la siguiente estructura de archivos:


    ```
    train/
        negative/
            0000.txt
            0001.txt
            ...
        positive/
            0000.txt
            0001.txt
            ...
        neutral/
            0000.txt
            0001.txt
            ...
    test/
        negative/
            0000.txt
            0001.txt
            ...
        positive/
            0000.txt
            0001.txt
            ...
        neutral/
            0000.txt
            0001.txt
            ...
    ```

    A partir de esta informacion escriba el código que permita generar
    dos archivos llamados "train_dataset.csv" y "test_dataset.csv". Estos
    archivos deben estar ubicados en la carpeta "output" ubicada en la raiz
    del repositorio.

    Estos archivos deben tener la siguiente estructura:

    * phrase: Texto de la frase. hay una frase por cada archivo de texto.
    * sentiment: Sentimiento de la frase. Puede ser "positive", "negative"
      o "neutral". Este corresponde al nombre del directorio donde se
      encuentra ubicado el archivo.

    Cada archivo tendria una estructura similar a la siguiente:

    ```
    |    | phrase                                                                                                                                                                 | target   |
    |---:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------|
    |  0 | Cardona slowed her vehicle , turned around and returned to the intersection , where she called 911                                                                     | neutral  |
    |  1 | Market data and analytics are derived from primary and secondary research                                                                                              | neutral  |
    |  2 | Exel is headquartered in Mantyharju in Finland                                                                                                                         | neutral  |
    |  3 | Both operating profit and net sales for the three-month period increased , respectively from EUR16 .0 m and EUR139m , as compared to the corresponding quarter in 2006 | positive |
    |  4 | Tampere Science Parks is a Finnish company that owns , leases and builds office properties and it specialises in facilities for technology-oriented businesses         | neutral  |
    ```


    """
    import os
    import zipfile

    import pandas as pd

    # Raíz del repositorio: un nivel arriba de homework/
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    input_zip = os.path.join(root, "files", "input.zip")
    extract_dir = os.path.join(root, "files")
    output_dir = os.path.join(root, "files", "output")

    # Descomprimir el archivo zip de entrada
    with zipfile.ZipFile(input_zip, "r") as z:
        z.extractall(extract_dir)

    # Crear la carpeta de salida si no existe
    os.makedirs(output_dir, exist_ok=True)

    def build_dataset(split):
        """Construye un DataFrame a partir de los archivos de texto de un split."""
        records = []
        base_path = os.path.join(extract_dir, "input", split)
        for sentiment in ["positive", "negative", "neutral"]:
            folder = os.path.join(base_path, sentiment)
            if not os.path.isdir(folder):
                continue
            for filename in os.listdir(folder):
                if filename.endswith(".txt"):
                    filepath = os.path.join(folder, filename)
                    with open(filepath, "r", encoding="utf-8") as f:
                        phrase = f.read().strip()
                    records.append({"phrase": phrase, "target": sentiment})
        return pd.DataFrame(records, columns=["phrase", "target"])

    # Generar y guardar train_dataset.csv
    train_df = build_dataset("train")
    train_df.to_csv(os.path.join(output_dir, "train_dataset.csv"), index=False)

    # Generar y guardar test_dataset.csv
    test_df = build_dataset("test")
    test_df.to_csv(os.path.join(output_dir, "test_dataset.csv"), index=False)
