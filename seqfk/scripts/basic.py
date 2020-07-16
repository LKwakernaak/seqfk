import analysis
import generate
from data import StoredxWriter

s = generate.main(
    prep_steps=int(1e6),
    samples=int(1e7),
    save_every=1,
    temp=1. / 3,
    writer=StoredxWriter
)

analysis.main()
