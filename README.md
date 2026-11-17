# Jak komputer uczy się rozpoznawać rysunki?

Materiały do warsztatów o sieciach konwolucyjnych dla liceum: notatnik Jupyter,
w którym mała sieć (LeNet-5) uczy się rozpoznawać rysunki z gry Quick, Draw!,
a Ty rysujesz myszą i sprawdzasz, co sieć o nich myśli.

Strona: <https://rodakt.github.io/obrazy-cnn>

## Jak uruchomić w domu

1. Zainstaluj **Miniforge** ze strony [conda-forge.org/download](https://conda-forge.org/download/)
   (Windows: zwykły instalator, wszystkie opcje domyślne).
2. Uruchom **Miniforge Prompt** z menu Start i wpisz: `mamba install -y numpy notebook`
   (macOS i Linux: to samo polecenie w terminalu).
3. Pobierz notatnik [`warsztaty_cnn.ipynb`](https://rodakt.github.io/obrazy-cnn/warsztaty_cnn.ipynb) do Pobranych.
4. W Miniforge Prompt wpisz `jupyter notebook`. Jupyter otworzy się w przeglądarce:
   wejdź do katalogu `Downloads` i kliknij notatnik.
5. Uruchamiaj komórki po kolei (**Shift+Enter**). Pierwsza pobierze resztę potrzebnych plików.

Następnym razem wystarczą kroki 4 i 5.

Bez internetu: pobierz całe repozytorium (**Code → Download ZIP**), rozpakuj
i otwórz notatnik w rozpakowanym katalogu.

## Licencje

- Kod (`warsztat.py`, komórki z kodem): [MIT](LICENSE).
- Tekst notatnika i strony: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- Rysunki w `dane/`: *Quick, Draw! Dataset*, Google Creative Lab,
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/),
  [github.com/googlecreativelab/quickdraw-dataset](https://github.com/googlecreativelab/quickdraw-dataset).
