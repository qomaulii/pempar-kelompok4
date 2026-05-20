from mpi4py import MPI
import csv

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

file = "AI_Impact_Student_Life_10K_2026.csv"

# ===============================
# MERGE SORT
# ===============================

def merge(left, right, col):
    merged = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i][col] >= right[j][col]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged

def merge_sort(data, col):
    if len(data) <= 1:
        return data
    mid   = len(data) // 2
    left  = merge_sort(data[:mid], col)
    right = merge_sort(data[mid:], col)
    return merge(left, right, col)

# ===============================
# LOAD CSV
# ===============================

def load_csv(filename):
    rows = []
    with open(filename, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tool       = row["Primary_AI_Tool"]
            gpa_before = float(row["GPA_Baseline"])
            gpa_after  = float(row["GPA_Post_AI"])
            diff       = gpa_after - gpa_before
            rows.append((tool, diff, gpa_before))
    return rows

# ===============================
# MAIN
# ===============================

def main():
    start = MPI.Wtime()
    print(f"Processor {rank} active of {size}")

    # ===============================
    # ROOT: Baca dan DIVIDE
    # ===============================
    if rank == 0:
        all_data  = load_csv(file)
        n         = len(all_data)
        part_size = n // size + (1 if n % size != 0 else 0)

        parts = []
        for i in range(size):
            begin = i * part_size
            end   = min(begin + part_size, n)
            parts.append(all_data[begin:end])

        print(f"Total data: {n}, part size: {part_size}")
    else:
        parts = None

    # ===============================
    # SCATTER
    # ===============================
    local_data = comm.scatter(parts, root=0)
    print(f"Processor {rank} received {len(local_data)} data")

    # ===============================
    # CONQUER — tiap prosesor sort data mentahnya
    # ===============================
    local_sorted = merge_sort(local_data, col=1)
    print(f"Processor {rank} done local merge sort")

    # ===============================
    # HITUNG AGREGASI LOKAL — sebelum gather
    # ===============================
    local_count = {}
    local_diff  = {}

    for tool, diff, gpa_before in local_sorted:
        local_count[tool] = local_count.get(tool, 0) + 1
        local_diff[tool]  = local_diff.get(tool, 0.0) + diff

    # ===============================
    # GATHER
    # ===============================
    all_sorted = comm.gather(local_sorted, root=0)
    all_counts = comm.gather(local_count, root=0)
    all_diffs  = comm.gather(local_diff, root=0)

    # ===============================
    # COMBINE — merge tree + gabung agregasi
    # ===============================
    if rank == 0:

        # Merge tree berpasangan
        step = 1
        while step < len(all_sorted):
            temp = []
            for i in range(0, len(all_sorted), 2 * step):
                if i + step < len(all_sorted):
                    merged = merge(all_sorted[i], all_sorted[i + step], col=1)
                    temp.append(merged)
                else:
                    temp.append(all_sorted[i])
            all_sorted = temp
            step *= 2

        # Gabung agregasi dari semua prosesor
        total_count = {}
        total_diff  = {}

        for d in all_counts:
            for tool in d:
                total_count[tool] = total_count.get(tool, 0) + d[tool]

        for d in all_diffs:
            for tool in d:
                total_diff[tool] = total_diff.get(tool, 0.0) + d[tool]

        # Sort hasil agregasi
        sorted_count  = merge_sort(list(total_count.items()), col=1)
        avg_list      = [(t, total_diff[t] / total_count[t]) for t in total_count]
        sorted_impact = merge_sort(avg_list, col=1)

        # ===============================
        # OUTPUT
        # ===============================
        print("\n==============================")
        print("MOST USED AI TOOL")
        print("==============================")
        for i, (tool, count) in enumerate(sorted_count, 1):
            print(f"{i}. {tool:<20} | Count: {count}")

        print("\n==============================")
        print("MOST IMPACTFUL AI TOOL TO GPA")
        print("==============================")
        for i, (tool, avg) in enumerate(sorted_impact, 1):
            print(f"{i}. {tool:<20} | Avg GPA: {avg:.3f}")

        end = MPI.Wtime()
        print(f"\nTotal execution time: {end - start:.4f} seconds")

if __name__ == "__main__":
    main()