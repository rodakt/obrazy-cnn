"""Kod pomocniczy do warsztatów CNN (prototyp).

Wymaga tylko: Python 3, numpy >= 1.20, Jupyter (IPython), tkinter.
Bez torcha, matplotlib i PIL — obrazki w notatniku są kodowane do PNG ręcznie.
"""
import base64, copy, os, struct, time, urllib.parse, urllib.request, zlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv

f32 = np.float32
NAZWY = {"apple": "jabłko", "house": "dom", "bicycle": "rower", "fish": "ryba", "star": "gwiazda",
         "cat": "kot", "dog": "pies", "car": "samochód", "tree": "drzewo", "umbrella": "parasol", "clock": "zegar"}
URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{}.npy"
# Strona z materiałami (GitHub Pages repozytorium obrazy-cnn). Zapasowe źródło danych:
# REPO + "/dane/<klasa>_<n>.part" — kopie plików z katalogu dane/. Puste = brak zapasu.
# Ten sam adres jest w pierwszej komórce notatnika (pobieranie warsztat.py).
REPO = "https://rodakt.github.io/obrazy-cnn"


# ---------------------------------------------------------------- dane

class Dane:
    def __init__(s, klasy, X, y, n_test):
        s.klasy = klasy; s.nazwy = [NAZWY.get(k, k) for k in klasy]
        s.Xte, s.yte = X[:n_test], y[:n_test]
        s.Xtr, s.ytr = X[n_test:], y[n_test:]


def pobierz(klasy, n=2000, n_test_na_klase=500, katalog="dane", seed=0):
    """Pobiera pierwsze n rysunków każdej klasy (HTTP Range, bez pełnych plików)."""
    os.makedirs(katalog, exist_ok=True)
    rozmiar = 128 + n*784  # z zapasem: nagłówek .npy ma zwykle 80 B, ale bywa dłuższy

    def jedna(k):
        p = os.path.join(katalog, f"{k}_{n}.part")
        if not (os.path.exists(p) and os.path.getsize(p) == rozmiar):
            try:
                req = urllib.request.Request(URL.format(urllib.parse.quote(k)), headers={"Range": f"bytes=0-{rozmiar-1}"})
                with urllib.request.urlopen(req, timeout=20) as r: b = r.read()
                if len(b) != rozmiar: raise IOError(f"pobrano {len(b)} B zamiast {rozmiar}")
            except Exception as e1:  # źródło główne zawiodło — próbujemy kopii w repozytorium
                if not REPO: raise IOError(f"Nie udało się pobrać danych „{k}”: {e1}") from e1
                try:
                    with urllib.request.urlopen(f"{REPO}/dane/{urllib.parse.quote(k)}_{n}.part", timeout=20) as r: b = r.read()
                    if len(b) != rozmiar: raise IOError(f"pobrano {len(b)} B zamiast {rozmiar}")
                except Exception as e2:
                    raise IOError(f"Nie udało się pobrać danych „{k}” ani ze źródła głównego ({e1}), "
                                  f"ani z zapasowego ({e2}).") from e2
            with open(p + ".tmp", "wb") as f: f.write(b)
            os.replace(p + ".tmp", p)
        with open(p, "rb") as f: b = f.read()
        h = 10 + int.from_bytes(b[8:10], "little")  # długość nagłówka .npy (zwykle 80 B łącznie)
        return np.frombuffer(b[h:h+n*784], np.uint8).reshape(n, 28, 28)

    t0 = time.time()
    with ThreadPoolExecutor(len(klasy)) as ex: obrazy = list(ex.map(jedna, klasy))
    X = np.concatenate(obrazy); y = np.repeat(np.arange(len(klasy)), n)
    # test: po n_test_na_klase z każdej klasy, reszta do treningu; całość wymieszana
    rng = np.random.default_rng(seed)
    idx = np.concatenate([rng.permutation(np.arange(i*n, (i+1)*n)) for i in range(len(klasy))]).reshape(len(klasy), n)
    te, tr = idx[:, :n_test_na_klase].ravel(), idx[:, n_test_na_klase:].ravel()
    te, tr = rng.permutation(te), rng.permutation(tr)
    o = np.concatenate([te, tr])
    d = Dane(list(klasy), X[o], y[o], len(te))
    print(f"Gotowe w {time.time()-t0:.1f} s: {len(klasy)} klas ({', '.join(d.nazwy)}), "
          f"{len(d.ytr)} rysunków do nauki, {len(d.yte)} do sprawdzania.")
    return d


# ---------------------------------------------------------------- sieć (LeNet-5 w numpy)

def _uni(rng, shape, fan): b = 1/np.sqrt(fan); return rng.uniform(-b, b, shape).astype(f32)

class Conv:
    def __init__(s, rng, cin, cout, k=5, pad=0):
        s.k, s.pad = k, pad; fan = cin*k*k; s.p = [_uni(rng, (fan, cout), fan), _uni(rng, (cout,), fan)]
        s.opis = f"splot {k}×{k}, {cout} filtrów"
    def fwd(s, x):
        if s.pad: x = np.pad(x, ((0, 0), (s.pad,)*2, (s.pad,)*2, (0, 0)))
        B, H, W, C = s.xs = x.shape; k = s.k; s.Ho, s.Wo = H-k+1, W-k+1
        s.cols = swv(x, (k, k), axis=(1, 2)).reshape(B*s.Ho*s.Wo, C*k*k)
        return (s.cols @ s.p[0] + s.p[1]).reshape(B, s.Ho, s.Wo, -1)
    def bwd(s, g):
        g2 = g.reshape(-1, g.shape[3]); s.g = [s.cols.T @ g2, g2.sum(0)]
        dc = (g2 @ s.p[0].T).reshape(s.xs[0], s.Ho, s.Wo, s.xs[3], s.k, s.k)
        dx = np.zeros(s.xs, f32)
        for i in range(s.k):
            for j in range(s.k): dx[:, i:i+s.Ho, j:j+s.Wo, :] += dc[..., i, j]
        return dx[:, s.pad:-s.pad, s.pad:-s.pad, :] if s.pad else dx

class ReLU:
    p = []; opis = "ReLU"
    def fwd(s, x): s.m = x > 0; return x*s.m
    def bwd(s, g): return g*s.m

class Pool:
    p = []; opis = "max-pooling 2×2"
    def fwd(s, x):
        B, H, W, C = s.xs = x.shape; r = x.reshape(B, H//2, 2, W//2, 2, C); m = r.max(axis=(2, 4))
        s.m = r == m[:, :, None, :, None, :]; return m
    def bwd(s, g): return (s.m*g[:, :, None, :, None, :]).reshape(s.xs)

class Flat:
    p = []; opis = "spłaszczenie"
    def fwd(s, x): s.xs = x.shape; return x.reshape(len(x), -1)
    def bwd(s, g): return g.reshape(s.xs)

class Lin:
    def __init__(s, rng, i, o): s.p = [_uni(rng, (i, o), i), _uni(rng, (o,), i)]; s.opis = f"warstwa gęsta, {o} neuronów"
    def fwd(s, x): s.x = x; return x @ s.p[0] + s.p[1]
    def bwd(s, g): s.g = [s.x.T @ g, g.sum(0)]; return g @ s.p[0].T


class LeNet:
    def __init__(s, n_klas, seed=None):
        """seed=None: za każdym razem inne losowe wagi (każdy uczeń ma swoją sieć)."""
        r = np.random.default_rng(seed); s.rng = r
        s.warstwy = [Conv(r, 1, 6, pad=2), ReLU(), Pool(), Conv(r, 6, 16), ReLU(), Pool(), Flat(),
                     Lin(r, 400, 120), ReLU(), Lin(r, 120, 84), ReLU(), Lin(r, 84, n_klas)]
        s.L = [l for l in s.warstwy if l.p]; s.t = 0
        s.poczatek = [[p.copy() for p in l.p] for l in s.L]   # wagi sprzed uczenia — do pokaz_mapy
        s.M = [[np.zeros_like(p) for p in l.p] for l in s.L]; s.V = [[np.zeros_like(p) for p in l.p] for l in s.L]

    def _fwd(s, x):
        for l in s.warstwy: x = l.fwd(x)
        return x

    def prawdopodobienstwa(s, X):
        """X: (N,28,28) uint8 albo float 0..1 -> (N, n_klas)."""
        X = _wejscie(X); out = []
        for i in range(0, len(X), 500):
            z = s._fwd(X[i:i+500]); z = np.exp(z - z.max(1, keepdims=True)); out.append(z/z.sum(1, keepdims=True))
        return np.concatenate(out)

    def krok(s, X, y, lr=1e-3, b1=.9, b2=.999, eps=1e-8):
        z = s._fwd(X); z = np.exp(z - z.max(1, keepdims=True)); z /= z.sum(1, keepdims=True)
        strata = -np.log(z[np.arange(len(y)), y] + 1e-12).mean()
        z[np.arange(len(y)), y] -= 1; g = z/len(y)
        for l in reversed(s.warstwy): g = l.bwd(g)
        s.t += 1; a = lr*np.sqrt(1-b2**s.t)/(1-b1**s.t)
        for l, M, V in zip(s.L, s.M, s.V):
            for p, gp, m, v in zip(l.p, l.g, M, V):
                m *= b1; m += (1-b1)*gp; v *= b2; v += (1-b2)*gp*gp; p -= a*m/(np.sqrt(v)+eps)
        return strata


def _wejscie(X):
    X = np.asarray(X)
    if X.ndim == 2: X = X[None]
    X = X.astype(f32)
    if X.max() > 1.5: X = X/255
    return X[..., None]


def opisz(model):
    print("Warstwy sieci:")
    for l in model.warstwy: print("  ", l.opis)
    n = sum(p.size for l in model.L for p in l.p)
    print(f"Liczba parametrów (wag) do nauczenia: {n:,}".replace(",", " "))


def trenuj(model, dane, epoki=5, batch=128, wykres=True):
    """Uczy sieć. W notatniku rysuje na żywo wykres straty i trafności (wykres=False wyłącza)."""
    Xtr = _wejscie(dane.Xtr); na_epoke = -(-len(dane.ytr)//batch); K = len(dane.klasy)
    Xpod, ypod = dane.Xte[:500], dane.yte[:500]      # podzbiór testu do szybkiego pomiaru w trakcie uczenia
    def pomiar(): return float((model.prawdopodobienstwa(Xpod).argmax(1) == ypod).mean())
    uchwyt = None
    if wykres and _w_notatniku():
        from IPython.display import HTML, display
        straty_w, traf_w = [], [(0, pomiar())]
        rysuj = lambda: HTML(_svg_uczenie(straty_w, traf_w, epoki*na_epoke, na_epoke, K))
        uchwyt = display(rysuj(), display_id=True)
    print(f"Uczenie: {epoki} epok × {len(dane.ytr)} rysunków")
    t0 = time.time()
    for e in range(epoki):
        q = model.rng.permutation(len(dane.ytr)); straty = []
        for i in range(0, len(q), batch):
            b = q[i:i+batch]; straty.append(model.krok(Xtr[b], dane.ytr[b]))
            if uchwyt:
                straty_w.append(float(straty[-1]))
                if len(straty_w) % 10 == 0 or len(straty) == na_epoke:
                    traf_w.append((len(straty_w), pomiar())); uchwyt.update(rysuj())
        acc = (model.prawdopodobienstwa(dane.Xte).argmax(1) == dane.yte).mean()
        print(f"  epoka {e+1}: strata {np.mean(straty):.3f}, trafność na rysunkach testowych {acc:.1%}"
              f"   ({time.time()-t0:.0f} s)", flush=True)


def _w_notatniku():
    """True w Jupyterze / Colabie (jest jądro), False w zwykłym Pythonie."""
    try:
        from IPython import get_ipython
        return hasattr(get_ipython(), "kernel")
    except Exception: return False


def _svg_uczenie(straty, traf, n_krokow, na_epoke, K):
    """Wykres uczenia jako SVG: po lewej strata w każdym kroku, po prawej trafność na rysunkach testowych."""
    W, H, l, r, t, b = 400, 240, 46, 14, 28, 36
    X = lambda i: l + i/max(n_krokow, 1)*(W - l - r)
    def panel(x0, tytul, ymax, znaczniki, tresc):
        Y = lambda v: t + (1 - min(max(v, 0), ymax)/ymax)*(H - t - b)
        o = [f'<g transform="translate({x0},0)"><text x="{l}" y="16" font-size="13" font-weight="bold" fill="#333">{tytul}</text>']
        for v, opis in znaczniki:
            o.append(f'<line x1="{l}" x2="{W-r}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="#e5e5e5"/>'
                     f'<text x="{l-6}" y="{Y(v)+4:.1f}" font-size="11" text-anchor="end" fill="#555">{opis}</text>')
        for e in range(n_krokow//na_epoke):
            if e: o.append(f'<line x1="{X(e*na_epoke):.1f}" x2="{X(e*na_epoke):.1f}" y1="{t}" y2="{H-b}" stroke="#ccc" stroke-dasharray="2,3"/>')
            o.append(f'<text x="{X((e+.5)*na_epoke):.1f}" y="{H-b+15}" font-size="11" text-anchor="middle" fill="#555">{e+1}</text>')
        o.append(f'<text x="{(l+W-r)/2}" y="{H-4}" font-size="11" text-anchor="middle" fill="#555">epoka</text>'
                 f'<rect x="{l}" y="{t}" width="{W-l-r}" height="{H-t-b}" fill="none" stroke="#888"/>')
        return "".join(o) + tresc(Y) + "</g>"
    def linia(pkt, Y, kolor, gr): return (f'<polyline fill="none" stroke="{kolor}" stroke-width="{gr}" stroke-linejoin="round" points="'
                                          + " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in pkt) + '"/>') if len(pkt) > 1 else ""
    def koniec(pkt, Y, kolor, opis):  # kropka i wartość na końcu linii
        if not pkt: return ""
        i, v = pkt[-1]; lewo = X(i) > W - r - 50
        return (f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="3.5" fill="{kolor}"/><text x="{X(i) + (-7 if lewo else 7):.1f}" y="{Y(v)-7:.1f}" '
                f'font-size="12" font-weight="bold" text-anchor="{"end" if lewo else "start"}" fill="{kolor}">{opis(v)}</text>')
    def p_strata(Y):
        gladka, m = [], 0.0
        for i, v in enumerate(straty): m = .9*m + .1*v; gladka.append((i+1, m/(1 - .9**(i+1))))   # średnia krocząca
        return (linia([(i+1, v) for i, v in enumerate(straty)], Y, "#b9cde5", 1) + linia(gladka, Y, "#2166ac", 2.2)
                + koniec(gladka, Y, "#2166ac", lambda v: f"{v:.2f}".replace(".", ",")))
    def p_traf(Y):
        return (f'<line x1="{l}" x2="{W-r}" y1="{Y(1/K):.1f}" y2="{Y(1/K):.1f}" stroke="#c62828" stroke-dasharray="5,4"/>'
                f'<text x="{W-r-4}" y="{Y(1/K)+13:.1f}" font-size="11" text-anchor="end" fill="#c62828">zgadywanie na ślepo</text>'
                + linia(traf, Y, "#1a7f37", 2.2) + koniec(traf, Y, "#1a7f37", lambda v: f"{v:.0%}"))
    ymax = max(1.0, np.ceil(np.log(K)*1.15*2)/2)
    zn_s = [(v, f"{v:g}".replace(".", ",")) for v in np.arange(0, ymax + .01, .5)]
    zn_t = [(v, f"{v:.0%}") for v in (0, .25, .5, .75, 1)]
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{2*W}" height="{H}" style="background:#fff;font-family:sans-serif">'
            + panel(0, "strata (im mniej, tym lepiej)", ymax, zn_s, p_strata)
            + panel(W, "trafność na rysunkach testowych", 1, zn_t, p_traf) + "</svg>")


# ---------------------------------------------------------------- wyświetlanie w notatniku (bez matplotlib)

def _png(img, skala=4):
    """Obraz uint8 -> PNG (base64), powiększony. 2D: tusz 0..255, czarne kreski na białym tle;
    3D (H, W, 3): gotowe kolory RGB."""
    a = np.asarray(img, np.uint8)
    if a.ndim == 2: a = 255 - a
    a = np.repeat(np.repeat(a, skala, 0), skala, 1)
    h, w = a.shape[:2]
    raw = b"".join(b"\x00" + a[i].tobytes() for i in range(h))
    def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0 if a.ndim == 2 else 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    return base64.b64encode(png).decode()

def _img(img, skala=4, podpis="", kolor="#333"):
    return (f'<div style="display:inline-block;margin:4px;text-align:center;font-size:12px;color:{kolor}">'
            f'<img src="data:image/png;base64,{_png(img, skala)}" style="border:1px solid #ccc;image-rendering:pixelated">'
            + (f'<br>{podpis}' if podpis else '') + '</div>')

def _html(s):
    from IPython.display import HTML, display
    display(HTML(s))


def _siatka(dane, ile, ile_test, model=None):
    """Przykłady z każdej kategorii: rysunki do nauki | rysunki testowe.
    Bez modelu testowe są zasłonięte kłódką; z modelem — odkryte, z odpowiedzią sieci."""
    td = 'padding:2px 8px;vertical-align:middle'
    reszta = lambda n: f'<span style="color:#888;font-size:12px;white-space:nowrap">… {n}</span>'
    zaslona = ('<div style="display:inline-block;margin:4px;width:58px;height:58px;line-height:58px;text-align:center;'
               'background:#e6e6e6;border:1px solid #ccc;font-size:22px">🔒</div>')
    naglowek_test = "testowe — odkryte 🔓" if model else "testowe — schowane przed siecią 🔒"
    w = [f'<tr><td></td><td style="{td};color:#2166ac;font-weight:bold">do nauki</td>'
         f'<td style="{td};border-left:2px dashed #aaa;color:#666;font-weight:bold">{naglowek_test}</td></tr>']
    for k, nazwa in enumerate(dane.nazwy):
        itr, ite = np.flatnonzero(dane.ytr == k), np.flatnonzero(dane.yte == k)
        nauka = "".join(_img(dane.Xtr[i], 2) for i in itr[:ile])
        if model is None: test = zaslona*min(ile_test, len(ite))
        else:
            P = model.prawdopodobienstwa(dane.Xte[ite[:ile_test]]); test = ""
            for i, p in zip(ite[:ile_test], P):
                j = p.argmax(); test += _img(dane.Xte[i], 2, f"{dane.nazwy[j]} {p[j]:.0%}", "#1a7f37" if j == k else "#c62828")
        w.append(f'<tr><td style="{td};font-weight:bold">{nazwa}</td><td style="{td}">{nauka}{reszta(len(itr))}</td>'
                 f'<td style="{td};border-left:2px dashed #aaa">{test}{reszta(len(ite))}</td></tr>')
    ntr, nte = len(dane.ytr), len(dane.yte)
    pasek = (f'<div style="display:flex;width:520px;margin:10px 0 0 8px;font-size:12px;color:#fff;font-weight:bold">'
             f'<div style="flex:{ntr};background:#2166ac;padding:4px 8px">{ntr} do nauki</div>'
             f'<div style="flex:{nte};background:{"#1a7f37" if model else "#888"};padding:4px 8px;white-space:nowrap">'
             f'{nte} testowych {"🔓" if model else "🔒"}</div></div>')
    return '<table style="border-collapse:collapse">' + "".join(w) + "</table>" + pasek


def pokaz_przyklady(dane, ile=6, ile_test=2):
    """Przykładowe rysunki każdej kategorii; testowe zasłonięte — sieć ich nie zobaczy podczas nauki."""
    _html(_siatka(dane, ile, ile_test))


def odkryj(model, dane, ile=6, ile_test=2):
    """Te same przykłady co w pokaz_przyklady, ale rysunki testowe odkryte — z odpowiedzią sieci."""
    _html(_siatka(dane, ile, ile_test, model)
          + '<p>Zielone — sieć zgadła, czerwone — pomyliła się.</p>')


def pokaz_test(model, dane, ile=15, seed=None):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dane.yte), ile, replace=False)
    P = model.prawdopodobienstwa(dane.Xte[idx])
    s = []
    for i, p in zip(idx, P):
        k = p.argmax(); ok = k == dane.yte[i]
        s.append(_img(dane.Xte[i], 3, f"{dane.nazwy[k]} {p[k]:.0%}" + ("" if ok else f"<br>(to: {dane.nazwy[dane.yte[i]]})"),
                      "#1a7f37" if ok else "#c62828"))
    acc = (model.prawdopodobienstwa(dane.Xte).argmax(1) == dane.yte).mean()
    _html("".join(s) + f"<p>Trafność na wszystkich {len(dane.yte)} rysunkach testowych: <b>{acc:.1%}</b></p>")


def _klasa(dane, nazwa):
    """Numer klasy po nazwie polskiej albo angielskiej; None, gdy nie podano; -1, gdy nie ma takiej."""
    if nazwa is None: return None
    n = str(nazwa).strip().lower()
    for lista in (dane.nazwy, dane.klasy):
        if n in lista: return lista.index(n)
    print("Nie znam kategorii „" + str(nazwa) + "”. Dostępne: " + ", ".join(dane.nazwy)); return -1


def macierz_pomylek(model, dane):
    """Tabela: wiersz — co było na rysunku testowym, kolumna — co odpowiedziała sieć."""
    K = len(dane.klasy); odp = model.prawdopodobienstwa(dane.Xte).argmax(1)
    M = np.zeros((K, K), int); np.add.at(M, (dane.yte, odp), 1)
    td = 'padding:4px 10px;text-align:center;border:1px solid #ccc'
    w = [f'<tr><td style="{td};border:none"></td><td colspan="{K}" style="{td};border:none;color:#666">sieć odpowiedziała:</td></tr>'
         f'<tr><td style="{td};border:none;color:#666">na rysunku było:</td>' + "".join(f'<th style="{td}">{n}</th>' for n in dane.nazwy)
         + f'<th style="{td};border:none;color:#666">trafność</th></tr>']
    for i in range(K):
        kom = []; razem = max(M[i].sum(), 1)
        for j in range(K):
            u = M[i, j]/razem
            tlo = f"rgba(26,127,55,{0.45*u:.2f})" if i == j else f"rgba(198,40,40,{min(1, 8*u):.2f})"
            opis = (f"{dane.nazwy[i]} rozpoznane poprawnie" if i == j else f"{dane.nazwy[i]} wzięte za: {dane.nazwy[j]}") + f" — {M[i, j]} z {razem}"
            kom.append(f'<td title="{opis}" style="{td};background:{tlo};color:#000">{M[i, j]}</td>')
        w.append(f'<tr><th style="{td};text-align:left">{dane.nazwy[i]}</th>' + "".join(kom)
                 + f'<td style="{td};border:none">{M[i, i]/razem:.0%}</td></tr>')
    poza = M.copy(); np.fill_diagonal(poza, 0); i, j = np.unravel_index(poza.argmax(), poza.shape)
    podsum = (f"Najczęstsza pomyłka: <b>{dane.nazwy[i]}</b> wzięte za: <b>{dane.nazwy[j]}</b> ({poza[i, j]} razy)." if poza.max()
              else "Sieć nie pomyliła się ani razu.")
    _html('<table style="border-collapse:collapse;font-size:13px">' + "".join(w) + "</table>"
          f'<p>Na przekątnej (zielone) trafienia, poza nią (czerwone) pomyłki — razem {poza.sum()} na {M.sum()} rysunków. {podsum}</p>')


def pokaz_pomylki(model, dane, ile=12, bylo=None, odpowiedz=None):
    """Rysunki testowe, przy których sieć pomyliła się, będąc najbardziej pewną swego.
    bylo / odpowiedz: nazwy kategorii, żeby zawęzić, np. bylo="ryba", odpowiedz="rower"."""
    kb, ko = _klasa(dane, bylo), _klasa(dane, odpowiedz)
    if kb == -1 or ko == -1: return
    P = model.prawdopodobienstwa(dane.Xte); odp = P.argmax(1); pewnosc = P.max(1)
    zle = odp != dane.yte
    if kb is not None: zle &= dane.yte == kb
    if ko is not None: zle &= odp == ko
    idx = np.flatnonzero(zle); idx = idx[np.argsort(-pewnosc[idx])][:ile]
    if not len(idx): print("Nie ma takich pomyłek."); return
    _html("".join(_img(dane.Xte[i], 3, f"{dane.nazwy[odp[i]]} {pewnosc[i]:.0%}<br>(to: {dane.nazwy[dane.yte[i]]})", "#c62828") for i in idx)
          + f'<p>Pomyłek tego rodzaju: {zle.sum()}. Pokazane te, przy których sieć była najbardziej pewna swojej (błędnej) odpowiedzi.</p>')


def rozpoznaj(model, rysunki, dane):
    """Pokazuje, jak model widzi rysunek (28×28) i co o nim myśli."""
    if rysunki is None or len(rysunki) == 0: print("Brak rysunku."); return
    rysunki = np.asarray(rysunki)
    if rysunki.ndim == 2: rysunki = rysunki[None]
    P = model.prawdopodobienstwa(rysunki)
    s = []
    for img, p in zip(rysunki, P):
        paski = "".join(
            f'<div style="font-size:13px;margin:2px 0"><span style="display:inline-block;width:80px">{dane.nazwy[k]}</span>'
            f'<span style="display:inline-block;width:{max(1, round(200*p[k]))}px;height:12px;background:{"#1a7f37" if k == p.argmax() else "#999"}"></span>'
            f' {p[k]:.0%}</div>' for k in np.argsort(-p))
        s.append(f'<div style="display:flex;align-items:center;gap:16px;margin:8px 0">{_img(img, 5, "tak widzi to sieć")}<div>{paski}</div></div>')
    _html("".join(s))


# ---------------------------------------------------------------- oszukaj sieć: bazgroły i obroty

def _bazgrol(rng):
    """Losowy bazgrół: 1–3 kreski, każda to błądzenie z bezwładnością (płynnie skręca)."""
    kreski = []
    for _ in range(rng.integers(1, 4)):
        p, kat = rng.uniform(0, 150, 2), rng.uniform(0, 2*np.pi); k = [p]
        for _ in range(rng.integers(8, 40)):
            kat += rng.normal(0, 0.6); p = p + 10*np.array([np.cos(kat), np.sin(kat)]); k.append(p)
        kreski.append(np.array(k))
    return kreski


def bazgroly(model, dane, ile=12, seed=None, n=200):
    """Losowe bazgroły (żadna z kategorii) i odpowiedzi sieci. Statystyka liczona na n bazgrołach, pokazane pierwsze ile."""
    rng = np.random.default_rng(seed)
    B = np.array([rasteryzuj(_bazgrol(rng)) for _ in range(max(n, ile))])
    P = model.prawdopodobienstwa(B); odp, pew = P.argmax(1), P.max(1)
    pew_test = model.prawdopodobienstwa(dane.Xte[:500]).max(1)
    ile_k = np.bincount(odp, minlength=len(dane.klasy)); k = ile_k.argmax()
    obrazki = "".join(_img(B[i], 3, f"{dane.nazwy[odp[i]]} {pew[i]:.0%}", "#c62828" if pew[i] > .9 else "#333") for i in range(ile))
    _html(obrazki + f'<p>Średnia pewność sieci na {len(B)} losowych bazgrołach: <b>{pew.mean():.0%}</b> '
          f'(na prawdziwych rysunkach testowych: {pew_test.mean():.0%}). '
          f'Pewność powyżej 90% (na czerwono): <b>{(pew > .9).sum()} z {len(B)}</b>. '
          f'Najczęstsza odpowiedź: <b>{dane.nazwy[k]}</b> ({ile_k[k]} z {len(B)}).</p>')


def _obroc_kreski(kreski, kat):
    """Obrót kresek o kat stopni przeciwnie do ruchu wskazówek zegara (oś y w dół, jak na ekranie)."""
    a = np.deg2rad(kat); c, s = np.cos(a), np.sin(a)
    return [np.asarray(k, float) @ np.array([[c, -s], [s, c]]) for k in kreski]


def _obroc_obraz(img, kat, S=4):
    """Obrót rysunku 28×28 i ponowne kadrowanie jak w Quick, Draw! (dłuższy bok 26 px), z wygładzaniem.
    Dla rysunków bez kresek (np. z danych); rysunki z okna obracamy po kreskach."""
    img = np.asarray(img, f32)/255
    a = np.deg2rad(kat); c, s = np.cos(a), np.sin(a)
    rot = lambda x, y: (c*x + s*y, -s*x + c*y)          # ten sam kierunek co _obroc_kreski
    ys, xs = np.nonzero(img > .3)
    if not len(ys): return np.zeros((28, 28), np.uint8)
    rx, ry = rot(xs + .5 - 14, ys + .5 - 14)
    lo = np.array([rx.min(), ry.min()]) - .5; roz = np.array([rx.max(), ry.max()]) + .5 - lo
    skala = 26/max(roz.max(), 1e-9); przes = 1 + (26 - roz*skala)/2
    v, u = (np.mgrid[0:28*S, 0:28*S] + .5)/S                 # punkty siatki wyjściowej
    qx, qy = (u - przes[0])/skala + lo[0], (v - przes[1])/skala + lo[1]
    sx, sy = c*qx - s*qy + 14 - .5, s*qx + c*qy + 14 - .5     # obrót odwrotny -> współrzędne w obrazie źródłowym
    x0, y0 = np.floor(sx).astype(int), np.floor(sy).astype(int); fx, fy = sx - x0, sy - y0
    P = np.pad(img, 1)
    def w(yy, xx):
        ok = (xx >= -1) & (xx <= 28) & (yy >= -1) & (yy <= 28)
        return np.where(ok, P[np.clip(yy + 1, 0, 29), np.clip(xx + 1, 0, 29)], 0)
    out = ((1-fx)*(1-fy)*w(y0, x0) + fx*(1-fy)*w(y0, x0+1) + (1-fx)*fy*w(y0+1, x0) + fx*fy*w(y0+1, x0+1))
    out = out.reshape(28, S, 28, S).mean((1, 3)).ravel()
    # interpolacja rozmywa kreskę; przywracamy jasności oryginału (te same wartości, nowe miejsca),
    # żeby sieć widziała różnicę w kształcie, a nie w ostrości
    wynik = np.zeros(28*28, np.uint8); wynik[np.argsort(-out, kind="stable")] = np.sort(np.round(255*img).ravel())[::-1]
    return wynik.reshape(28, 28)


def _obroty(img, katy):
    """Obrócone wersje rysunku: po kreskach, jeśli rysunek pochodzi z okna, inaczej jako obraz."""
    kreski = next((k for im, k in reversed(_kreski) if np.array_equal(im, img)), None)
    if kreski is not None: return np.array([rasteryzuj(_obroc_kreski(kreski, k)) for k in katy])
    return np.array([_obroc_obraz(img, k) for k in katy])


def _svg_obrot(katy, linie):
    """Wykres: oś x — kąt obrotu, oś y — pewność 0..100%. linie: [(opis, kolor, przerywana, wartości)]."""
    W, H, l, r, t, b = 580, 250, 46, 175, 14, 36
    X = lambda k: l + k/max(katy[-1], 1)*(W - l - r); Y = lambda v: t + (1 - v)*(H - t - b)
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" style="background:#fff;font-family:sans-serif">']
    for v in (0, .25, .5, .75, 1):
        o.append(f'<line x1="{l}" x2="{W-r}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="#e5e5e5"/>'
                 f'<text x="{l-6}" y="{Y(v)+4:.1f}" font-size="11" text-anchor="end" fill="#555">{v:.0%}</text>')
    for k in katy[::2]:
        o.append(f'<text x="{X(k):.1f}" y="{H-b+15}" font-size="11" text-anchor="middle" fill="#555">{k}°</text>')
    o.append(f'<text x="{(l+W-r)/2}" y="{H-4}" font-size="11" text-anchor="middle" fill="#555">obrót</text>'
             f'<rect x="{l}" y="{t}" width="{W-l-r}" height="{H-t-b}" fill="none" stroke="#888"/>')
    for i, (opis, kolor, przer, war) in enumerate(linie):
        o.append(f'<polyline fill="none" stroke="{kolor}" stroke-width="2.2" stroke-linejoin="round"'
                 + (' stroke-dasharray="6,4"' if przer else '') + ' points="'
                 + " ".join(f"{X(k):.1f},{Y(v):.1f}" for k, v in zip(katy, war)) + '"/>'
                 + "".join(f'<circle cx="{X(k):.1f}" cy="{Y(v):.1f}" r="2.5" fill="{kolor}"/>' for k, v in zip(katy, war) if not przer))
        y = t + 10 + 20*i
        o.append(f'<line x1="{W-r+12}" x2="{W-r+34}" y1="{y}" y2="{y}" stroke="{kolor}" stroke-width="2.2"'
                 + (' stroke-dasharray="6,4"' if przer else '') + f'/><text x="{W-r+40}" y="{y+4}" font-size="12" fill="#333">{opis}</text>')
    return "".join(o) + "</svg>"


def obracaj(model, img, dane, klasa="ryba", krok=15, do=180, ile_test=300):
    """Obraca rysunek co krok stopni (0..do) i pokazuje odpowiedzi sieci; dla porównania średnia po rysunkach
    testowych tej samej kategorii, obróconych tak samo."""
    img = _jeden(img); kl = _klasa(dane, klasa)
    if img is None: print("Brak rysunku."); return
    if kl is None or kl == -1: return
    katy = list(range(0, do + 1, krok))
    R = _obroty(img, katy); P = model.prawdopodobienstwa(R); odp = P.argmax(1)
    T = dane.Xte[dane.yte == kl][:ile_test]
    PT = np.stack([model.prawdopodobienstwa(np.array([_obroc_obraz(x, k) for x in T]))[:, kl].mean() for k in katy])
    inne = [j for j in range(len(dane.klasy)) if j != kl]; rywal = max(inne, key=lambda j: P[:, j].sum())
    obrazki = "".join(_img(r, 2, f"{k}°<br>{dane.nazwy[j]} {p[j]:.0%}", "#1a7f37" if j == kl else "#c62828")
                      for r, k, p, j in zip(R, katy, P, odp))
    nazwa = dane.nazwy[kl]
    linie = [(f"{nazwa} — Twój rysunek", "#1a7f37", False, P[:, kl]),
             (f"{dane.nazwy[rywal]} — Twój rysunek", "#c62828", False, P[:, rywal]),
             (f"{nazwa} — testowe, średnio", "#1a7f37", True, PT)]
    pierwszy = next((k for k, j in zip(katy, odp) if j != kl), None)
    if pierwszy is None: podsum = f"Sieć przy każdym obrocie widzi kategorię „{nazwa}”."
    elif pierwszy == 0: podsum = (f"Już bez obrotu sieć nie widzi tu kategorii „{nazwa}” "
                                  f"— narysuj ją wyraźniej i uruchom komórkę jeszcze raz.")
    else: podsum = f"Pierwszy obrót, przy którym sieć nie widzi już kategorii „{nazwa}”: <b>{pierwszy}°</b>."
    _html(f'<div>{obrazki}</div>{_svg_obrot(katy, linie)}'
          f'<p>{podsum} Linia przerywana: średnia pewność „{nazwa}” dla {len(T)} rysunków testowych tej kategorii, '
          f'obróconych tak samo (czerwona linia — kategoria, która najczęściej „przejmuje” Twój rysunek).</p>')


def pokaz_liczby(img):
    """Rysunek jako liczby: (przycięty rysunek) -> 28×28 pikseli -> tabela 784 liczb 0..255."""
    img = np.asarray(img)
    if img.ndim == 3: img = img[0]
    panele = []
    kreski = next((k for im, k in reversed(_kreski) if np.array_equal(im, img)), None)
    if kreski is not None:  # rysunek z okna: pokaż też wersję przed zmniejszeniem
        panele.append(_img(rasteryzuj(kreski, R=3.9, S=6, duzy=True), 1, "twój rysunek, przycięty"))
    panele.append(_img(img, 6, "28 × 28 pikseli"))
    wiersze = []
    for r in range(28):
        kom = []
        for c in range(28):
            v = int(img[r, c])
            kom.append(f'<td title="wiersz {r+1}, kolumna {c+1}: {v}" style="background:rgb({255-v},{255-v},{255-v});'
                       f'color:{"#fff" if v > 120 else ("#bbb" if v == 0 else "#000")}">{v}</td>')
        wiersze.append("<tr>" + "".join(kom) + "</tr>")
    tabela = ('<style>.wl-tab td{width:17px;height:13px;padding:0;text-align:center;border:1px solid #eee;'
              'font:8px monospace}</style><table class="wl-tab" style="border-collapse:collapse">' + "".join(wiersze) + "</table>")
    strzalka = '<div style="font-size:28px;color:#888">→</div>'
    _html(f'<div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap">{strzalka.join(panele)}{strzalka}{tabela}</div>'
          f'<p>{img.size} liczb. Kreska: {(img > 0).sum()} pikseli (w tym {(img == 255).sum()} pełnych = 255), '
          f'tło: {(img == 0).sum()} pikseli = 0.</p>')


# ---------------------------------------------------------------- splot i „co sieć widzi”

FILTRY = {  # ręczne filtry 3×3 do pokaz_splot; rysunki to cienkie kreski, więc szukamy kresek, nie krawędzi
    "pionowe":  [[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]],
    "poziome":  [[-1, -1, -1], [2, 2, 2], [-1, -1, -1]],
    "ukośne":   [[2, -1, -1], [-1, 2, -1], [-1, -1, 2]],
    "ukośne2":  [[-1, -1, 2], [-1, 2, -1], [2, -1, -1]],
    "rozmycie": [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
}
_NIEB, _CZERW, _RAMKA = np.array([33, 102, 172.]), np.array([178, 24, 43.]), (255, 140, 0)
_LEGENDA = ('<span style="background:rgb(33,102,172);color:#fff;padding:0 6px">niebieski</span> — liczby dodatnie, '
            '<span style="background:rgb(178,24,43);color:#fff;padding:0 6px">czerwony</span> — ujemne, biały — zero')

def _barwy(M, vmax=None):
    """Tablica liczb -> kolory RGB: dodatnie niebieskie, ujemne czerwone, zero białe."""
    M = np.asarray(M, float); vmax = vmax or max(np.abs(M).max(), 1e-9)
    t = np.clip(M/vmax, -1, 1)[..., None]
    return np.round(255 + np.where(t >= 0, t*(_NIEB - 255), -t*(_CZERW - 255))).astype(np.uint8)

def _powieksz(a, S): return np.repeat(np.repeat(a, S, 0), S, 1)

def _ramka(a, r0, r1, c0, c1, S, g=2):
    """Rysuje ramkę wokół komórek [r0..r1] × [c0..c1] na obrazie RGB powiększonym S razy."""
    H, W = a.shape[:2]
    y0, y1, x0, x1 = max(r0*S, 0), min((r1+1)*S, H), max(c0*S, 0), min((c1+1)*S, W)
    a[y0:y0+g, x0:x1] = a[y1-g:y1, x0:x1] = a[y0:y1, x0:x0+g] = a[y0:y1, x1-g:x1] = _RAMKA

def _liczba(v): return f"{v:g}".replace("-", "−")

def _tabela(M, kolory, rozm=26):
    """Mała tabela liczb z zadanymi kolorami tła (RGB na komórkę)."""
    w = []
    for wiersz, kw in zip(np.asarray(M), kolory):
        w.append("<tr>" + "".join(
            f'<td style="width:{rozm}px;height:{rozm}px;padding:0;text-align:center;border:1px solid #ccc;font:11px monospace;'
            f'background:rgb({k[0]},{k[1]},{k[2]});color:{"#fff" if int(k[0])+int(k[1])+int(k[2]) < 380 else "#000"}">{_liczba(v)}</td>'
            for v, k in zip(wiersz, kw)) + "</tr>")
    return '<table style="border-collapse:collapse;display:inline-table">' + "".join(w) + "</table>"

def _jeden(img):
    """Jeden rysunek 28×28 z tego, co zwraca narysuj() albo z danych; None, gdy pusto."""
    img = np.asarray(img)
    if img.size == 0: return None
    return img[0] if img.ndim == 3 else img


def pokaz_splot(img, filtr="pionowe", miejsce=None):
    """Splot rysunku z małym filtrem: rysunek ∗ filtr → wynik, plus rachunek dla jednego miejsca.
    filtr: nazwa z FILTRY albo własna kwadratowa tabelka liczb (np. 3×3);
    miejsce: (wiersz, kolumna) liczone od 1 — domyślnie tam, gdzie wynik jest największy."""
    img = _jeden(img)
    if img is None: print("Brak rysunku."); return
    if isinstance(filtr, str):
        klucz = filtr.strip().lower().replace("ukosne", "ukośne")
        if klucz not in FILTRY: print("Nie znam filtra „" + filtr + "”. Dostępne: " + ", ".join(FILTRY)); return
        F = np.array(FILTRY[klucz], float); nazwa = f"filtr „{klucz}”"
    else:
        F = np.array(filtr, float); nazwa = "własny filtr"
        if F.ndim != 2 or F.shape[0] != F.shape[1] or F.shape[0] % 2 == 0 or F.shape[0] > 7:
            print("Filtr musi być kwadratową tabelką 3×3, 5×5 albo 7×7."); return
    k = len(F); p = k//2
    okna = swv(np.pad(img.astype(float), p), (k, k))   # (28, 28, k, k): fragment rysunku wokół każdego piksela
    wynik = (okna*F).sum((2, 3))
    if miejsce is None: r, c = np.unravel_index(wynik.argmax(), wynik.shape)
    else: r, c = (int(np.clip(m - 1, 0, 27)) for m in miejsce)
    S = 6
    A = _powieksz(np.repeat((255 - img)[..., None], 3, 2).astype(np.uint8), S); _ramka(A, r-p, r+p, c-p, c+p, S)
    B = _powieksz(_barwy(wynik), S); _ramka(B, r, r, c, c, S)
    znak = lambda t: f'<div style="font-size:28px;color:#888">{t}</div>'
    tabF = f'<div style="text-align:center;font-size:12px;color:#333">{_tabela(F, _barwy(F))}<br>{nazwa}</div>'
    fragment = okna[r, c]; szare = np.repeat((255 - fragment)[..., None], 3, 2).astype(np.uint8)
    skladniki = [f"{_liczba(a)}·{'(' + _liczba(b) + ')' if b < 0 else _liczba(b)}"
                 for a, b in zip(fragment.ravel(), F.ravel()) if a and b]
    rachunek = (" + ".join(skladniki) + " = " if 0 < len(skladniki) <= 9 else "") + f"<b>{_liczba(wynik[r, c])}</b>"
    _html(f'<div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap">{_img(A, 1, "rysunek")}{znak("∗")}{tabF}'
          f'{znak("→")}{_img(B, 1, "wynik splotu")}</div>'
          f'<p>Wynik to znowu tabela 28 × 28 liczb, od {_liczba(wynik.min())} do {_liczba(wynik.max())}: {_LEGENDA}.</p>'
          f'<p><b>Jak powstaje jedna liczba wyniku</b> — miejsce w pomarańczowej ramce (wiersz {r+1}, kolumna {c+1}):</p>'
          f'<div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap">'
          f'<div style="text-align:center;font-size:12px;color:#333">{_tabela(fragment, szare)}<br>fragment rysunku</div>{znak("×")}{tabF}'
          f'{znak("→")}<div style="font-size:13px">mnożymy liczby leżące na tych samych miejscach i dodajemy:<br>{rachunek}</div></div>')


def pokaz_filtry(model, img):
    """Co sieć widzi: filtry pierwszej warstwy splotowej i to, co każdy z nich znajduje na rysunku,
    a niżej odpowiedzi drugiej warstwy splotowej."""
    img = _jeden(img)
    if img is None: print("Brak rysunku."); return
    W = model.warstwy; k = W[0].k
    F = W[0].p[0].T.reshape(-1, k, k)                  # (6, 5, 5)
    a1, a2 = _mapy(model, img)
    fmax, m1, m2 = np.abs(F).max(), max(a1.max(), 1e-9), max(a2.max(), 1e-9)
    kol = "".join(
        f'<div style="display:inline-block;text-align:center;margin:0 4px">{_img(_powieksz(_barwy(F[i], fmax), 14), 1, f"filtr {i+1}")}<br>'
        f'<span style="color:#888">↓</span><br>{_img(_powieksz(_barwy(a1[:, :, i], m1), 3), 1)}</div>' for i in range(len(F)))
    mapy2 = "".join(_img(_powieksz(_barwy(a2[:, :, i], m2), 5), 1) for i in range(a2.shape[2]))
    _html(f'<div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap">{_img(img, 5, "rysunek")}<div>{kol}</div></div>'
          f'<p>U góry {len(F)} filtrów {k}×{k} pierwszej warstwy ({_LEGENDA}). Pod każdym — miejsca rysunku, '
          f'w których ten filtr znalazł swój wzorzec (im ciemniej, tym mocniej).</p>'
          f'<p><b>Druga warstwa splotowa</b>: {a2.shape[2]} map {a2.shape[0]}×{a2.shape[1]}. {_O_WARSTWIE2}</p><div>{mapy2}</div>')


_O_WARSTWIE2 = ("Jej filtry nie patrzą już na rysunek, tylko na mapy z pierwszej warstwy — "
                "szukają wzorców złożonych z prostszych wzorców.")

def _mapy(model, img):
    """Mapy aktywacji (po ReLU) obu warstw splotowych dla jednego rysunku: (28, 28, 6), (10, 10, 16)."""
    W = model.warstwy
    a1 = W[1].fwd(W[0].fwd(_wejscie(img)))             # (1, 28, 28, 6)
    a2 = W[4].fwd(W[3].fwd(W[2].fwd(a1)))              # (1, 10, 10, 16) — po poolingu i drugim splocie
    return a1[0], a2[0]

def _na_poczatku(model):
    """Kopia sieci z wagami sprzed uczenia."""
    m = copy.deepcopy(model)
    for l, p0 in zip(m.L, m.poczatek):
        for p, q in zip(l.p, p0): p[...] = q
    return m


def pokaz_mapy(model, img):
    """Co znajdują warstwy splotowe na rysunku — przed uczeniem (losowe wagi) i po uczeniu."""
    img = _jeden(img)
    if img is None: print("Brak rysunku."); return
    przed, po = _mapy(_na_poczatku(model), img), _mapy(model, img)
    def wiersz(a, S, opis):  # każdy wiersz we własnej skali (po uczeniu liczby bywają większe)
        m = max(a.max(), 1e-9)
        return (f'<div style="display:flex;align-items:center"><span style="display:inline-block;width:110px;font-size:12px;color:#333">{opis}</span>'
                + "".join(_img(_powieksz(_barwy(a[:, :, i], m), S), 1) for i in range(a.shape[2])) + "</div>")
    def blok(nr, S, tytul):
        return (f'<p style="margin:10px 0 2px"><b>{tytul}</b></p>'
                + wiersz(przed[nr], S, "przed uczeniem") + wiersz(po[nr], S, "po uczeniu"))
    a1, a2 = po
    _html(f'<div style="display:flex;align-items:flex-start;gap:16px;flex-wrap:wrap">{_img(img, 5, "rysunek")}<div>'
          + blok(0, 3, f"Pierwsza warstwa splotowa: {a1.shape[2]} map {a1.shape[0]}×{a1.shape[1]}")
          + blok(1, 5, f"Druga warstwa splotowa: {a2.shape[2]} map {a2.shape[0]}×{a2.shape[1]}")
          + '</div></div>'
          f'<p>Każda mapa pokazuje, gdzie na rysunku jeden filtr znalazł swój wzorzec — im ciemniej, tym mocniej. '
          f'Mapa w wierszu „po uczeniu” leży pod mapą tego samego filtra sprzed uczenia. '
          f'Druga warstwa: {_O_WARSTWIE2[0].lower() + _O_WARSTWIE2[1:]}</p>')


# ---------------------------------------------------------------- schemat sieci

_ETAPY = {  # kolor ramki, kolor tła panelu
    "wejscie": ("#555", "#f2f2f2"), "splot": ("#2b6cb0", "#e6f0fa"), "pool": ("#c05621", "#fdf0e4"),
    "gesta": ("#2f855a", "#e5f4ea"), "wyjscie": ("#6b46c1", "#efe9fb")}

def _hex(rgb): return "#%02x%02x%02x" % tuple(int(v) for v in rgb)

def pokaz_siec(model, img, dane=None):
    """Kolorowy schemat sieci: rysunek przechodzi przez kolejne warstwy, a w każdej widać,
    co z niego zostało (prawdziwe mapy i neurony tej sieci dla tego rysunku)."""
    img = _jeden(img)
    if img is None: print("Brak rysunku."); return
    _html(_svg_siec(model, img, dane))

def _svg_siec(model, img, dane):
    x, akt = _wejscie(img), []
    for l in model.warstwy: x = l.fwd(x); akt.append(x[0])
    z = np.exp(akt[-1] - akt[-1].max()); p = z/z.sum()
    nazwy = dane.nazwy if dane is not None else [f"kategoria {k+1}" for k in range(len(p))]
    H, cy, top, dol = 360, 165, 34, 278
    el, linie, przod, x0 = [], [], [], 12  # tło (panele, mapy), linie połączeń, neurony

    def panel(w, etap, tytul, opis, strzalka=True):
        nonlocal x0
        kol, tlo = _ETAPY[etap]; cx = x0 + w/2
        el.append(f'<rect x="{x0}" y="{top}" width="{w}" height="{dol-top}" rx="12" fill="{tlo}" stroke="{kol}" stroke-width="1.5"/>'
                  f'<text x="{cx}" y="{top+20}" text-anchor="middle" font-weight="bold" font-size="14" fill="{kol}">{tytul}</text>'
                  + "".join(f'<text x="{cx}" y="{dol+18+15*i}" text-anchor="middle" font-size="12" fill="#333">{t}</text>'
                            for i, t in enumerate(opis)))
        x0 += w + 30
        if strzalka: el.append(f'<path d="M{x0-27},{cy} h18" stroke="#888" stroke-width="2" marker-end="url(#gr)"/>')
        return cx

    def stos(a, S, d, etap, cx):  # mapy (H, W, C) jako talia kart; najmocniejsza na wierzchu
        a = a[:, :, np.argsort(-a.sum((0, 1)))]
        n, s = a.shape[2], a.shape[0]*S; w = s + (n-1)*d
        m = max(np.percentile(a[a > 0], 95) if (a > 0).any() else 0, 1e-9)  # ostrzej niż max: mapy nie są blade
        xl, yf = cx - w/2, cy + w/2 - s
        for i in reversed(range(n)):
            obraz = _barwy(a[:, :, i], m) if etap != "wejscie" else 255 - np.repeat(a[:, :, i, None]*255, 3, 2)
            el.append(f'<image x="{xl+i*d}" y="{yf-i*d}" width="{s}" height="{s}" style="image-rendering:pixelated" '
                      f'href="data:image/png;base64,{_png(_powieksz(obraz.astype(np.uint8), S), 1)}"/>'
                      f'<rect x="{xl+i*d}" y="{yf-i*d}" width="{s}" height="{s}" fill="none" stroke="{_ETAPY[etap][0]}" stroke-width="1"/>')

    def kolumna(v, ile, cx, r=7):  # wybrane neurony warstwy gęstej, kolor = siła odpowiedzi
        idx = np.linspace(0, len(v)-1, ile).round().astype(int); m = max(v.max(), 1e-9)
        ys = cy + (np.arange(ile) - (ile-1)/2)*19
        for k, yy in zip(idx, ys):
            przod.append(f'<circle cx="{cx}" cy="{yy:.1f}" r="{r}" fill="{_hex(_barwy(v[k], m))}" stroke="#2f855a" stroke-width="1.2"/>')
        przod.append(f'<text x="{cx}" y="{ys[-1]+24:.1f}" text-anchor="middle" font-size="12" fill="#2f855a">{len(v)}</text>')
        return [(cx, yy) for yy in ys]

    def polacz(A, B):
        for xa, ya in A:
            for xb, yb in B: linie.append(f'<line x1="{xa:.1f}" y1="{ya:.1f}" x2="{xb:.1f}" y2="{yb:.1f}"/>')

    cx = panel(110, "wejscie", "Rysunek", ["28 × 28 = 784 liczby"]); stos(_wejscie(img)[0], 3, 0, "wejscie", cx)
    cx = panel(150, "splot", "Splot", ["6 filtrów szuka", "prostych wzorców"]); stos(akt[1], 3, 7, "splot", cx)
    cx = panel(110, "pool", "Zmniejszanie", ["2 × mniejsze"]); stos(akt[2], 3, 5, "pool", cx)
    cx = panel(130, "splot", "Splot", ["16 filtrów szuka", "wzorców z wzorców"]); stos(akt[4], 5, 3, "splot", cx)
    cx = panel(110, "pool", "Zmniejszanie", ["2 × mniejsze"]); stos(akt[5], 5, 2.5, "pool", cx)
    cx = panel(140, "gesta", "Warstwy gęste", ["każdy neuron patrzy", "na wszystko naraz"], strzalka=False)
    A = kolumna(akt[8], 9, cx - 30); B = kolumna(akt[10], 7, cx + 30)
    polacz(A, B)
    kol = _ETAPY["wyjscie"][0]; x_wy = x0 + 22
    K = len(p); slowo = "kategorie" if K % 10 in (2, 3, 4) and K % 100 not in (12, 13, 14) else "kategorii"
    panel(150, "wyjscie", "Odpowiedź", [f"{K} {slowo}, razem 100%"], strzalka=False)
    ys = cy + 6 + (np.arange(K) - (K-1)/2)*min(30, 180/max(K-1, 1)); r = 9 if K <= 6 else 6
    porz = np.argsort(-p)
    C = [(x_wy, yy) for yy in ys]; polacz(B, C)
    for (xx, yy), k in zip(C, porz):
        naj = k == porz[0]
        przod.append(f'<circle cx="{xx}" cy="{yy:.1f}" r="{r}" fill="{_hex(255 - p[k]*(255 - np.array([107, 70, 193.])))}" stroke="{kol}" stroke-width="1.5"/>'
                  f'<text x="{xx+16}" y="{yy-2:.1f}" font-size="12" fill="#222" font-weight="{"bold" if naj else "normal"}">{nazwy[k]}</text>'
                  f'<rect x="{xx+16}" y="{yy+3:.1f}" width="{max(1, 70*p[k]):.1f}" height="5" fill="{kol if naj else "#bbb"}"/>'
                  f'<text x="{xx+92}" y="{yy+8:.1f}" font-size="11" fill="#555">{p[k]:.0%}</text>')
    W = x0 - 30 + 12
    n = f"{sum(q.size for l in model.L for q in l.p):,}".replace(",", " ")
    el.append(f'<text x="{W/2}" y="{H-8}" text-anchor="middle" font-size="13" fill="#333">'
              f'Liczb (wag) do nauczenia: <tspan font-weight="bold">{n}</tspan> — w filtrach i w warstwach gęstych</text>')
    return (f'<svg viewBox="0 0 {W} {H}" style="width:100%;max-width:{W}px;font-family:sans-serif">'
          '<defs><marker id="gr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">'
          '<path d="M0,0 L10,5 L0,10 z" fill="#888"/></marker></defs>'
          + "".join(el) + f'<g stroke="#2f855a" stroke-opacity=".25" stroke-width="1">{"".join(linie)}</g>'
          + "".join(przod) + "</svg>")


def _resnet152():
    """Warstwy ResNet-152: splot 7×7, pooling, 50 bloków po 3 sploty (3+8+36+3), pooling, warstwa wyjściowa."""
    return ["splot", "pool"] + ["splot"]*150 + ["pool", "wyjscie"]

_SIECI = [  # nazwa, rok i opis, warstwy (te z wagami i pooling), co ile warstw skrót
    ("nasza sieć (LeNet-5)", "1998 · 5 warstw z wagami · 61 tys. wag",
     ["splot", "pool", "splot", "pool", "gesta", "gesta", "wyjscie"], 0),
    ("AlexNet", "2012 · 8 warstw z wagami · 60 mln wag",
     ["splot", "pool", "splot", "pool", "splot", "splot", "splot", "pool", "gesta", "gesta", "wyjscie"], 0),
    ("ResNet-152", "2015 · 152 warstwy z wagami · 60 mln wag", _resnet152(), 3),
]

def pokaz_skale():
    """Te same klocki, większa skala: LeNet, AlexNet i ResNet-152 narysowane w jednej skali —
    każdy prostokąt to jedna warstwa (kolory jak w pokaz_siec)."""
    _html(_svg_skale())

def _svg_skale():
    wb, wp, gap, h, lewo = 4.5, 2.5, 1.5, 44, 250
    el, y = [], 20
    for nazwa, opis, warstwy, skrot in _SIECI:
        y += 34 if skrot else 14
        el.append(f'<text x="10" y="{y+h/2-3}" font-size="14" font-weight="bold" fill="#222">{nazwa}</text>'
                  f'<text x="10" y="{y+h/2+14}" font-size="12" fill="#555">{opis}</text>')
        x, xs = lewo, []
        for t in warstwy:
            w = wp if t == "pool" else wb
            el.append(f'<rect x="{x:.1f}" y="{y}" width="{w}" height="{h}" rx="1" fill="{_ETAPY[t][0]}"/>')
            xs.append((x, w, t)); x += w + gap
        if skrot:  # łuk nad każdym blokiem trzech splotów: sygnał przeskakuje blok
            sp = [(xx, w) for xx, w, t in xs if t == "splot"][1:]
            for i in range(0, len(sp), skrot):
                a, b = sp[i][0], sp[i+skrot-1][0] + sp[i+skrot-1][1]
                el.append(f'<path d="M{a:.1f},{y-1} Q{(a+b)/2:.1f},{y-16} {b:.1f},{y-1}" fill="none" stroke="#d69e2e" stroke-width="1.5"/>')
        y += h + 10
    W = lewo + 160*(wb+gap)
    leg = [("splot", "splot"), ("pool", "zmniejszanie"), ("gesta", "warstwa gęsta"), ("wyjscie", "odpowiedź")]
    x = 10; y += 22
    for t, op in leg:
        el.append(f'<rect x="{x}" y="{y-11}" width="12" height="12" rx="2" fill="{_ETAPY[t][0]}"/>'
                  f'<text x="{x+18}" y="{y}" font-size="13" fill="#333">{op}</text>'); x += 40 + 8*len(op)
    el.append(f'<path d="M{x},{y-2} Q{x+12},{y-16} {x+24},{y-2}" fill="none" stroke="#d69e2e" stroke-width="1.5"/>'
              f'<text x="{x+32}" y="{y}" font-size="13" fill="#333">skrót — sygnał przeskakuje kilka warstw</text>')
    H = y + 12
    return (f'<svg viewBox="0 0 {W:.0f} {H}" style="width:100%;max-width:{W:.0f}px;font-family:sans-serif">'
            + "".join(el) + "</svg>")


# ---------------------------------------------------------------- rysowanie myszą (tkinter)

_kreski = []  # (obraz 28×28, kreski) dla rysunków z okna — do pokaz_liczby

def rasteryzuj(kreski, R=2.6, S=4, duzy=False):
    """Kreski [[(x,y),...],...] -> obraz 28×28 uint8 w stylu Quick, Draw!:
    dłuższy bok zajmuje 26 px (margines 1 px), krótszy wyśrodkowany, kreska ok. 2 px
    (zmierzone na danych). Rysujemy w siatce S razy gęstszej i uśredniamy (wygładzanie).
    duzy=True: zwraca tę gęstszą siatkę (28S × 28S) bez zmniejszania — do pokazania."""
    kreski = [np.asarray(k, float) for k in kreski if len(k)]
    if not kreski: return np.zeros((28, 28), np.uint8)
    pts = np.concatenate(kreski); lo = pts.min(0); rozp = pts.max(0) - lo
    N = 28*S; m = S + R; skala = (N - 2*m)/max(rozp.max(), 1e-9)
    przes = m + ((N - 2*m) - rozp*skala)/2
    yy, xx = np.mgrid[0:N, 0:N] + 0.5; P = np.stack([xx.ravel(), yy.ravel()], 1)
    tusz = np.zeros(N*N, bool)
    for k in kreski:
        k = (k - lo)*skala + przes
        if len(k) == 1: k = np.vstack([k, k])
        A, B = k[:-1], k[1:]
        for a, b in zip(A, B):  # odległość punktów siatki od odcinka a-b
            ab = b - a; L2 = ab @ ab
            sel = ~tusz
            q = P[sel] - a
            t = np.clip((q @ ab)/L2, 0, 1) if L2 > 0 else 0
            d2 = ((q - np.outer(t, ab) if L2 > 0 else q)**2).sum(1)
            tusz[np.flatnonzero(sel)[d2 <= R*R]] = True
    if duzy: return (255*tusz).astype(np.uint8).reshape(N, N)
    img = tusz.reshape(28, S, 28, S).mean((1, 3))
    return np.round(255*img).astype(np.uint8)


def narysuj(ile=1, rozmiar=400, tytul="Narysuj"):
    """Otwiera okno do rysowania. Zwraca tablicę (ile, 28, 28) — tyle rysunków, ile zatwierdzono."""
    import tkinter as tk
    wynik, kreski = [], []
    root = tk.Tk(); root.title(tytul); root.resizable(False, False)
    info = tk.Label(root, font=("Arial", 14)); info.pack(pady=6)
    cv = tk.Canvas(root, width=rozmiar, height=rozmiar, bg="white", cursor="pencil",
                   highlightthickness=1, highlightbackground="#888"); cv.pack(padx=10)
    def odswiez(): info.config(text=f"Rysunek {len(wynik)+1} z {ile}" if ile > 1 else "Narysuj myszą i kliknij „Gotowe”")
    def start(e): kreski.append([(e.x, e.y)])
    def ruch(e):
        if not kreski: return
        x0, y0 = kreski[-1][-1]; cv.create_line(x0, y0, e.x, e.y, width=8, capstyle="round", fill="black")
        kreski[-1].append((e.x, e.y))
    def wyczysc(): kreski.clear(); cv.delete("all")
    def gotowe():
        if not kreski: return  # pusty rysunek — ignorujemy
        wynik.append(rasteryzuj(kreski)); _kreski.append((wynik[-1], [list(k) for k in kreski])); wyczysc()
        if len(wynik) >= ile: koniec()
        else: odswiez()
    def koniec(): root.quit(); root.destroy()
    cv.bind("<ButtonPress-1>", start); cv.bind("<B1-Motion>", ruch)
    przyciski = tk.Frame(root); przyciski.pack(pady=8)
    tk.Button(przyciski, text="Wyczyść", font=("Arial", 12), command=wyczysc).pack(side="left", padx=6)
    tk.Button(przyciski, text="Gotowe", font=("Arial", 12), command=gotowe).pack(side="left", padx=6)
    if ile > 1: tk.Button(przyciski, text="Zakończ", font=("Arial", 12), command=koniec).pack(side="left", padx=6)
    root.protocol("WM_DELETE_WINDOW", koniec)
    odswiez()
    # okno na wierzch (w Windows bywa otwierane za przeglądarką)
    root.attributes("-topmost", True); root.lift(); root.focus_force()
    root.after(500, lambda: root.attributes("-topmost", False))
    root.mainloop()
    return np.array(wynik, np.uint8).reshape(-1, 28, 28)
