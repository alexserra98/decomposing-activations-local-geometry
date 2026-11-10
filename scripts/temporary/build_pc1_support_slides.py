"""Export the validated nine mean panels as a short scientific PDF slide deck."""
import json
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path('outputs/experiments/pc1_length_percentile_sweep/job_1932529')
OUT = ROOT / 'slides'
NAVY, TEAL, GRAY = '#23364A', '#007F86', '#526173'


def slide(title, subtitle, blocks):
    fig = plt.figure(figsize=(17, 8.8), facecolor='white')
    fig.text(.055, .90, title, fontsize=29, weight='bold', color=NAVY)
    fig.text(.055, .835, subtitle, fontsize=16, color=GRAY)
    for y, heading, body in blocks:
        fig.text(.065, y, heading, fontsize=21, weight='bold', color=TEAL)
        fig.text(.065, y-.055, body, fontsize=18, color=NAVY, va='top', linespacing=1.6)
    fig.text(.055, .035, 'DALG · PC1 support coverage · 10 novembre 2026', fontsize=11, color=GRAY)
    return fig


def main():
    OUT.mkdir(exist_ok=True)
    panels = [ROOT / f'm{m}_q{q}' / '03_pannello_medie.png'
              for m in (90, 95, 99) for q in (80, 90, 95)]
    assert all(p.is_file() for p in panels)
    path = OUT / 'PC1_supporto_kNN_sweep.pdf'
    with PdfPages(path) as pdf:
        intros = [
            ('PC1 segue il supporto delle attivazioni?',
             'Sensibilità alla lunghezza del segmento e alla soglia di accettazione kNN', [
                (.71, 'Domanda', 'Quanto del segmento lungo PC1 resta nel supporto empirico?\nQuanto supera direzioni casuali con lo stesso centro e la stessa lunghezza?'),
                (.49, 'Disegno: 9 combinazioni', 'Massa gaussiana centrale m = 90%, 95%, 99%  ×  percentile q = 80, 90, 95.\nSempre k = 20 vicini; solo PC1; HDDC-EM e KMeans+PCA, K = 50 e 100.'),
                (.27, 'Quattro casi per ogni pannello', 'Years e Temperature, layer 4 e 28. Le nove pagine di risultati riportano\nle medie per componente nel layout originale, con soglia di rango 0,1.')]),
            ('1 · Definire il supporto con kNN calibrato',
             'Stesso spazio PCA75, stessa popolazione e stessa calibrazione per tutti i segmenti', [
                (.72, 'Distanza e soglia', r'$s(x)=d_{20}(x,R)$     $\tau_q=\operatorname{quantile}_q\{s(z):z\in Q\}$'+'\nUna posizione è accettata se s(x) ≤ τq (quantile con metodo “higher”).'),
                (.49, 'Riferimento e calibrazione separati', 'Split deterministico 80% R / 20% Q, seed 42. Years: 1601 / 401 punti;\nTemperature: 1200 / 300. q è un percentile; τq è una distanza euclidea.'),
                (.27, 'Spazio di valutazione', 'PCA75 salvata, senza whitening: Years 141D / 28D; Temperature 4D / 3D\n(layer 4 / 28). PC1 originale proiettata nello spazio comune di ciascun caso.')]),
            ('2 · Segmento, copertura e controllo casuale',
             'm indica massa di probabilità gaussiana centrale, non varianza spiegata', [
                (.73, 'Lunghezza del segmento', r'$a_m=\Phi^{-1}((1+m)/2)\sqrt{\lambda_1}$'+'  (semilunghezza prima della proiezione).\nRispetto a m = 80%: fattori 1,2835 (90%), 1,5294 (95%), 2,0099 (99%).'),
                (.50, 'Due score da leggere insieme', r'$C_{PC1}=\frac{1}{N}\sum_j\mathbf{1}[s(x_j)\leq\tau_q]$'+'\nΔ = copertura PC1 − media delle coperture di 32 direzioni casuali (in pp).'),
                (.27, 'Controllo abbinato e precisione', 'Direzioni casuali isotrope, stesso centro e stessa lunghezza di PC1.\nGriglia finale: 4001 posizioni; errore massimo verificato < 0,10 pp.\nK è il numero di componenti del modello; k = 20 è il numero di vicini.')]),
        ]
        for title, subtitle, blocks in intros:
            fig = slide(title, subtitle, blocks)
            pdf.savefig(fig)
            if title.startswith('PC1'):
                fig.savefig(OUT / 'copertina.png', dpi=110)
            plt.close(fig)
        for source in panels:
            fig = plt.figure(figsize=(17, 8.8), facecolor='white')
            ax = fig.add_axes([0, 0, 1, 1])
            ax.imshow(plt.imread(source))
            ax.axis('off')
            pdf.savefig(fig, dpi=200)
            plt.close(fig)
        fig = slide('Lettura dei risultati e limiti',
                    'Un indicatore di supporto direzionale, condizionato alla geometria di ogni componente', [
            (.72, 'Copertura e Δ rispondono a domande diverse', 'Copertura alta: il segmento attraversa posizioni accettate. Δ positivo:\nPC1 supera le direzioni casuali abbinate. Un 100% può indicare saturazione.'),
            (.49, 'I segmenti più lunghi attenuano la saturazione', 'Nelle statistiche mediane separate: 31/144 coperture sotto 100%, contro\n0/48 nel controllo m = 80%; 50 Δ mediani positivi. Le tabelle qui mostrano medie.'),
            (.26, 'Interpretazione descrittiva', 'Centri e lunghezze dipendono dal modello: non è un confronto di sola direzione.\nValidation set ricostruito, già usato per selezione; nessun test di significatività.\nAudit indipendente superato: 576 riepiloghi e 43.074 voci componente/configurazione.')])
        pdf.savefig(fig)
        plt.close(fig)
    (OUT / 'manifest.json').write_text(json.dumps({
        'job_id': os.environ.get('SLURM_JOB_ID'), 'pdf': str(path), 'pages': 13,
        'panels': [str(p) for p in panels], 'aggregation': 'ordinary component means',
        'source_run': '1932529', 'independent_audit': '1932567',
        'method': 'PC1 only; k=20; m90/95/99 x q80/90/95',
    }, indent=2)+'\n')
    print(f'Created 13-slide deck: {path}', flush=True)


if __name__ == '__main__':
    main()
