import cv2
import numpy as np


def czyste_pca_2d(macierz_2d, k):
    """Wykonuje kompresję i natychmiastową dekompresję (rekonstrukcję)

    na dowolnej macierzy dwuwymiarowej za pomocą czystej matematyki.
    """
    # Zabezpieczenie: k nie może być większe niż liczba cech (kolumn)
    liczba_kolumn = macierz_2d.shape[1]
    if k > liczba_kolumn:
        k = liczba_kolumn

    # KROK 1: Centrowanie danych (odejmowanie średniej od każdej kolumny)
    srednia = np.mean(macierz_2d, axis=0)
    dane_wycentrowane = macierz_2d - srednia

    # KROK 2: Obliczenie macierzy kowariancji (wymiary: kolumny x kolumny)
    # rowvar=False oznacza, że kolumny to cechy, a wiersze to próbki
    macierz_kowariancji = np.cov(dane_wycentrowane, rowvar=False)

    # KROK 3: Wyznaczenie wartości i wektorów własnych
    # eigh jest zoptymalizowane dla symetrycznych macierzy kowariancji
    wartosci_wlasne, wektory_wlasne = np.linalg.eigh(macierz_kowariancji)

    # KROK 4: Sortowanie od największej wartości własnej (odwracamy indeksy)
    indeksy_posortowane = np.argsort(wartosci_wlasne)[::-1]
    wektory_wlasne = wektory_wlasne[:, indeksy_posortowane]

    # KROK 5: Wybór 'k' najważniejszych składowych (obcinamy macierz transformacji)
    W = wektory_wlasne[:, :k]

    # KROK 6: KOMPRESJA (Rzutowanie danych w nową przestrzeń)
    dane_skompresowane = np.dot(dane_wycentrowane, W)

    # KROK 7: DEKOMPRESJA / REKONSTRUKCJA
    # Mnożymy przez odwrócony klucz (transponowany W) i dodajemy średnią
    dane_zrekonstruowane = np.dot(dane_skompresowane, W.T) + srednia

    # Na koniec upewniamy się, że wartości mieszczą się w zakresie pikseli 0-255
    # i konwertujemy z powrotem na liczby całkowite (uint8)
    obraz_wynikowy = np.clip(dane_zrekonstruowane, 0, 255).astype(np.uint8)

    return obraz_wynikowy


def kompresja_globalna_3d(obraz, k):
    """PODEJŚCIE 1: Analizuje całe kolorowe zdjęcie na raz.

    Skleja kanały R, G, B w jedną wielką płaską macierz.
    """
    # Jeśli zdjęcie już jest szare (2D), leci standardowym PCA
    if len(obraz.shape) == 2:
        return czyste_pca_2d(obraz, k)

    wysokosc, szerokosc, kanaly = obraz.shape

    # Przekształcamy obraz z (H, W, 3) do (H, W * 3)
    # Czyli potrójnie rozciągamy wiersze w bok, łącząc bity kolorów obok siebie
    obraz_plaski = obraz.reshape(wysokosc, szerokosc * kanaly)

    # Wykonujemy PCA na tym "szerokim" obrazie
    obraz_zrekonstruowany_plaski = czyste_pca_2d(obraz_plaski, k)

    # Przywracamy oryginalne wymiary 3D (H, W, 3)
    return obraz_zrekonstruowany_plaski.reshape(wysokosc, szerokosc, kanaly)


def kompresja_kanal_po_kanale(obraz, k):
    """PODEJŚCIE 2: Rozbija zdjęcie na 3 osobne warstwy (R, G, B)

    i dla każdej z nich liczy PCA niezależnie, po czym składa je do kupy.
    """
    if len(obraz.shape) == 2:
        return czyste_pca_2d(obraz, k)

    wysokosc, szerokosc, kanaly = obraz.shape
    zrekonstruowane_kanaly = []

    # Iterujemy po każdym kanale osobno (w OpenCV kolejność to zazwyczaj B, G, R)
    for i in range(kanaly):
        pojedynczy_kanal = obraz[:, :, i]

        # Liczymy PCA dla jednej matrycy 500x500
        zrekonstruowany_kanal = czyste_pca_2d(pojedynczy_kanal, k)
        zreconstructed_kanaly.append(zrekonstruowany_kanal)

    # Składamy przetworzone kanały z powrotem w jeden obraz trójwymiarowy
    return np.stack(zreconstructed_kanaly, axis=2)


# --- PRZYKŁAD UŻYCIA ---
if __name__ == "__main__":
    # 1. Wczytaj swoje zdjęcie (podaj poprawną ścieżkę do pliku!)
    # Jeśli chcesz wymusić od razu szary obraz, odkomentuj flagę cv2.IMREAD_GRAYSCALE
    sciezka_do_zdjecia = "twoje_zdjecie.jpg"
    oryginal = cv2.imread(sciezka_do_zdjecia)

    if oryginal is None:
        print(
            f"Nie znaleziono pliku {sciezka_do_zdjecia}. Wygenerowano losową macierz do testu."
        )
        # Tworzymy sztuczny obrazek kolorowy 500x500x3 do testu, gdyby brakowało pliku
        oryginal = np.random.randint(
            0, 256, (500, 500, 3), dtype=np.uint8
        )

    # Pobieramy wymiary, żeby pomóc użytkownikowi wybrać limit 'k'
    if len(oryginal.shape) == 3:
        h, w, c = original.shape
        print(f"Wczytano obraz kolorowy o rozmiarze: {h}x{w} (3 kanały)")
        print(f"Maksymalne 'k' dla metody kanałowej: {w}")
        print(f"Maksymalne 'k' dla metody globalnej 3D: {w * c}")
    else:
        h, w = oryginal.shape
        print(f"Wczytano obraz w odcieniach szarości o rozmiarze: {h}x{w}")
        print(f"Maksymalne 'k': {w}")

    # 2. RĘCZNY WYBÓR LICZBY CECH (SKŁADOWYCH) - Tutaj wpisujesz ile chcesz!
    KOMPONENTY = 30

    # 3. Odpalenie obu metod kompresji i dekompresji
    wynik_globalny = kompresja_globalna_3d(oryginal, k=KOMPONENTY)
    wynik_kanalowy = kompresja_kanal_po_kanale(oryginal, k=KOMPONENTY)

    # 4. Zapisanie efektów pracy na dysk
    cv2.imwrite("wynik_globalny_3d.jpg", wynik_globalny)
    cv2.imwrite("wynik_kanalowy.jpg", wynik_kanalowy)
    print(
        f"Gotowe! Zdjęcia po dekompresji dla k={KOMPONENTY} zostały zapisane."
    )
