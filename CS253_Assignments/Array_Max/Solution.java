// THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING
// A TUTOR OR CODE WRITTEN BY OTHER STUDENTS - Dylan Parker

import java.io.*;
import java.util.*;
import java.util.StringTokenizer;
public class ArrayMax {

    // We will maintain a min-PQ of Entry objects.
    // Each Entry (i,v) represents an assignment "A[i]=v".
    static class Entry implements Comparable<Entry> {
        int i, v;
        Entry(int i, int v) { this.i=i; this.v=v; }
        // We negate the "v" comparison, so that PriorityQueue (a MinPQ)
        // returns the Entry with the maximum v.  We break ties with i,
        // so we can find the leftmost appearance of the maximum.

        public int compareTo(Entry that) {
            int dif = -(this.v - that.v);
            if (dif==0) // break ties with the index i
                dif = this.i - that.i;
            return dif;
        }
    }

    public static void main(String[] args) throws IOException{
        // Buffered output (for faster printing)
        PrintWriter out = new PrintWriter(System.out);
        // Buffered input (we also avoid java.util.Scanner)
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        StringTokenizer st = new StringTokenizer(br.readLine());
        int M = Integer.parseInt(st.nextToken()),
                N = Integer.parseInt(st.nextToken());
        int[] a = new int[M]; // initially all zero
        // Create MinPQ, and add an entry for a[0]=0 (that's all we need)
        IndexMinPQ<Entry> pq = new IndexMinPQ<>(M);
        pq.insert(0, new Entry(0, 0));
        // Loop through the N assignment lines
        for (int n=0; n<N; ++n) {
            // read the line, parse i and v
            st = new StringTokenizer(br.readLine());
            int i = Integer.parseInt(st.nextToken()),
                    v = Integer.parseInt(st.nextToken());
            // do the assignment in the array
            a[i] = v;
            // Add an Entry recording this assignment
            if(pq.contains(i)){
                pq.changeKey(i,new Entry(i,v));
            }else {
                pq.insert(i, new Entry(i, v));
            }
            // Get the head of the queue (Entry with maximum v value)
            Entry head = pq.minKey();
            // While the head is stale (no longer in the array), discard it.
            while (a[head.i] != head.v) {
                pq.delMin();        // discard it
                head = pq.minKey(); // the next head
            }
            // Report location of the largest value, a[head.i]==head.v
            out.println(head.i);
        }
        out.close();
    }
}

class IndexMinPQ<Key extends Comparable<Key>> implements Iterable<Integer> {
    private int maxN;        // maximum number of elements on PQ
    private int n;           // number of elements on PQ
    private int[] pq;        // binary heap using 1-based indexing
    private int[] qp;        // inverse of pq - qp[pq[i]] = pq[qp[i]] = i
    private Key[] keys;      // keys[i] = priority of i

    //initializes an empty priority queue
    public IndexMinPQ(int maxN) {
        if(maxN < 0) throw new IllegalArgumentException();
        this.maxN=maxN;
        n=0;
        keys = (Key[]) new Comparable[maxN + 1];
        pq = new int[maxN + 1];
        qp = new int[maxN + 1];
        for (int i = 0; i <= maxN; i++)
            qp[i] = -1;
    }

    //returns true if queue is empty
    public boolean isEmpty() {
        return n == 0;
    }

    //checks if queue contains i
    public boolean contains(int i) {
        validateIndex(i);
        return qp[i] != -1;
    }

    //returns size
    public int size() {
        return n;
    }

    //inserts key with index
    public void insert(int i, Key key) {
        validateIndex(i);
        if (contains(i)) throw new IllegalArgumentException("index is already in the priority queue");
        n++;
        qp[i] = n;
        pq[n] = i;
        keys[i] = key;
        swim(n);
    }

    //returns a minimum key
    public Key minKey() {
        if (n == 0) throw new NoSuchElementException("Priority queue underflow");
        return keys[pq[1]];
    }

    //removes a minimum key and returns its associated value
    public int delMin() {
        if (n == 0) throw new NoSuchElementException("Priority queue underflow");
        int min = pq[1];
        exch(1, n--);
        sink(1);
        assert min == pq[n+1];
        qp[min] = -1;        // delete
        keys[min] = null;    // to help with garbage collection
        pq[n+1] = -1;        // not needed
        return min;
    }

    //change the key with its associated index
    public void changeKey(int i, Key key) {
        validateIndex(i);
        if (!contains(i)) throw new NoSuchElementException("index is not in the priority queue");
        keys[i] = key;
        swim(qp[i]);
        sink(qp[i]);
    }

    //change the key with is associated index
    @Deprecated
    public void change(int i, Key key) {
        changeKey(i, key);
    }

    //remove key associated with index i
    public void delete(int i) {
        validateIndex(i);
        if (!contains(i)) throw new NoSuchElementException("index is not in the priority queue");
        int index = qp[i];
        exch(index, n--);
        swim(index);
        sink(index);
        keys[i] = null;
        qp[i] = -1;
    }

    // throw an IllegalArgumentException if i is an invalid index
    private void validateIndex(int i) {
        if (i < 0) throw new IllegalArgumentException("index is negative: " + i);
        if (i >= maxN) throw new IllegalArgumentException("index >= capacity: " + i);
    }

    //general helper functions
    private boolean greater(int i, int j) {
        return keys[pq[i]].compareTo(keys[pq[j]]) > 0;
    }

    private void exch(int i, int j) {
        int swap = pq[i];
        pq[i] = pq[j];
        pq[j] = swap;
        qp[pq[i]] = i;
        qp[pq[j]] = j;
    }

    //heap functions
    private void swim(int k) {
        while (k > 1 && greater(k/2, k)) {
            exch(k, k/2);
            k = k/2;
        }
    }

    private void sink(int k) {
        while (2*k <= n) {
            int j = 2*k;
            if (j < n && greater(j, j+1)) j++;
            if (!greater(k, j)) break;
            exch(k, j);
            k = j;
        }
    }

    //iterators
    public Iterator<Integer> iterator() { return new HeapIterator(); }

    private class HeapIterator implements Iterator<Integer> {
        // create a new pq
        private IndexMinPQ<Key> copy;

        // add all elements to copy of heap
        // takes linear time since already in heap order so no keys move
        public HeapIterator() {
            copy = new IndexMinPQ<Key>(pq.length - 1);
            for (int i = 1; i <= n; i++)
                copy.insert(pq[i], keys[pq[i]]);
        }

        public boolean hasNext()  { return !copy.isEmpty();                     }
        public void remove()      { throw new UnsupportedOperationException();  }

        public Integer next() {
            if (!hasNext()) throw new NoSuchElementException();
            return copy.delMin();
        }
    }
}