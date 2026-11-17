# Jak komputer uczy się rozpoznawać rysunki?

Materiały do warsztatów o sieciach konwolucyjnych dla liceum: notatnik Jupyter,
w którym mała sieć (LeNet-5) uczy się rozpoznawać rysunki z gry Quick, Draw!,
a Ty rysujesz myszą i sprawdzasz, co sieć o nich myśli.

Strona: <https://rodakt.github.io/obrazy-cnn>

## Jak uruchomić w domu

1. Zainstaluj Pythona 3 z [python.org](https://www.python.org/downloads/).
   Na Linuksie doinstaluj okienka do rysowania: `sudo apt install python3-tk`.
2. Zainstaluj pakiety: `pip install numpy jupyter`
3. Pobierz [`warsztaty_cnn.ipynb`](https://rodakt.github.io/obrazy-cnn/warsztaty_cnn.ipynb),
   w jego katalogu uruchom `jupyter notebook` i otwórz plik.
4. Uruchamiaj komórki po kolei (**Shift+Enter**). Pierwsza pobierze resztę potrzebnych plików.

Bez internetu: pobierz całe repozytorium (**Code → Download ZIP**), rozpakuj
i otwórz notatnik w rozpakowanym katalogu.

## Licencje

- Kod (`warsztat.py`, komórki z kodem): [MIT](LICENSE).
- Tekst notatnika i strony: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- Rysunki w `dane/`: *Quick, Draw! Dataset*, Google Creative Lab,
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/),
  [github.com/googlecreativelab/quickdraw-dataset](https://github.com/googlecreativelab/quickdraw-dataset).
