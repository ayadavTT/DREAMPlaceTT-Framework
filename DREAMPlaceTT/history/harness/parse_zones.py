import sys, csv, statistics, os
csvpath = sys.argv[1]
FREQ = 1.349937
open_z = {}; durs = {}
with open(csvpath, newline="") as f:
    r = csv.reader(f); next(r, None)
    for row in r:
        if len(row) < 14: continue
        core = (row[1].strip(), row[2].strip()); risc = row[3].strip()
        try: cyc = int(row[5])
        except: continue
        zone = row[10].strip(); typ = row[11].strip(); kfile = os.path.basename(row[13].strip())
        if not zone: continue
        key = (core, risc, zone)
        if typ == "ZONE_START": open_z[key] = cyc
        elif typ == "ZONE_END" and key in open_z:
            d = cyc - open_z.pop(key)
            if d >= 0: durs.setdefault((zone, risc, kfile), []).append(d)
rows = []
for (zone,risc,kf), ds in durs.items():
    if zone.endswith("-FW") or zone.endswith("-KERNEL") or "KERNEL-CONFIG" in zone: continue
    rows.append((kf, zone, risc, len(ds), statistics.median(ds)/FREQ/1000.0, max(ds)/FREQ/1000.0))
rows.sort(key=lambda x:(x[0], -x[4]))
hdr = ("kernel","zone","risc","n","med_us","max_us")
print("%-26s%-16s%-7s%6s%10s%10s" % hdr)
for kf,z,risc,n,med,mx in rows:
    print("%-26s%-16s%-7s%6d%10.2f%10.2f" % (kf,z,risc,n,med,mx))
