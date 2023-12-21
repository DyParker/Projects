//THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING

//A TUTOR OR CODE WRITTEN BY OTHER STUDENTS - DYLAN PARKER

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.Stack;
import java.util.stream.IntStream;


class Result {
    /*
     * Complete the 'addPrereq' function below.
     *
     * The function accepts following parameters:
     *  1. INTEGER prereq
     *  2. INTEGER course
     */

    private final int V;
    private final Bag<Integer>[] adj; //adjacency lists

    public Result(int V){
        this.V=V;
        //Creates empty digraph with V vertices
        adj = (Bag<Integer>[]) new Bag[V];
        for(int v=0; v<V; v++)
            adj[v] = new Bag<Integer>();
    }
    private void addEdge(int v, int w){
        adj[v].add(w);
    }
    public Iterable<Integer> adj(int v){
        return adj[v];
    }
    public int V(){
        return V;
    }

    //Add prerequisite
    public void addPrereq(int prereq, int course) {
        addEdge(prereq, course);
    }

    public boolean isPossible() {
        DirectedCycle finder= new DirectedCycle(this);

        return !finder.hasCycle();
    }

}
class Bag < Item > implements Iterable < Item > {
    private Item[] a;          //array of items
    private int N = 0;         //number of elements in the bag

    public Bag() {
        a = (Item[]) new Object[1];
    }

    public boolean isEmpty() { return N == 0; }
    public int size()        { return N;      }


    //resize
    private void resize(int max) {
        Item[] temp = (Item[]) new Object[max];
        for (int i = 0; i < N; i++)
            temp[i] = a[i];
        a = temp;
    }

    // insert an item
    public void add(Item item) {
        if (item == null) throw new NullPointerException();
        if (N == a.length) resize(2*a.length);  //double size of array if necessary
        a[N++] = item;
    }


    public Iterator < Item > iterator()  { return new MyIterator();}



    private class MyIterator implements Iterator {
        private int n ;    //next item to return

        public MyIterator() {
            n = N;
        }
        public boolean hasNext()  { return n > 0;}
        public void remove()      { throw new UnsupportedOperationException();}

        public Item next() {
            if (!hasNext()) throw new NoSuchElementException();
            return a[--n];
        }
    }
}
class DirectedCycle {
    private boolean[] marked;
    private int[] edgeTo;
    private boolean[] onStack;
    private Stack<Integer> cycle;

    //determines if G is a directed cycle
    public DirectedCycle(Result G) {
        marked  = new boolean[G.V()];
        onStack = new boolean[G.V()];
        edgeTo  = new int[G.V()];
        for (int v = 0; v < G.V(); v++)
            if (!marked[v] && cycle == null) dfs(G, v);
    }

    //run DFS and find a directed cycle (if one exists)
    private void dfs(Result G, int v) {
        onStack[v] = true;
        marked[v] = true;
        for (int w : G.adj(v)) {
            //if directed cycle found return
            if (cycle != null) return;

                //found new vertex, so recur
            else if (!marked[w]) {
                edgeTo[w] = v;
                dfs(G, w);
            }
            //trace back directed cycle
            else if (onStack[w]) {
                cycle = new Stack<Integer>();
                for (int x = v; x != w; x = edgeTo[x]) {
                    cycle.push(x);
                }
                cycle.push(w);
                cycle.push(v);
                assert check();
            }
        }
        onStack[v] = false;
    }

    //checks if there is a cycle
    public boolean hasCycle() {
        return cycle != null;
    }

    public Iterable<Integer> cycle() {
        return cycle;
    }

    //verifies cycle
    private boolean check() {
        if (hasCycle()) {
            //verify cycle
            int first = -1, last = -1;
            for (int v : cycle()) {
                if (first == -1) first = v;
                last = v;
            }
            if (first != last) {
                System.err.printf("cycle begins with %d and ends with %d\n", first, last);
                return false;
            }
        }
        return true;
    }
}

public class Solution {
    public static void main(String[] args) throws IOException {
        BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(System.in));
        String[] firstMultipleInput = bufferedReader.readLine().replaceAll("\\s+$", "").split(" ");

        int numCourses = Integer.parseInt(firstMultipleInput[0]);
        int numPrereqs = Integer.parseInt(firstMultipleInput[1]);

        Result result = new Result(numCourses);
        IntStream.range(0, numPrereqs).forEach(numPrereqsItr -> {
            try {
                String[] secondMultipleInput = bufferedReader.readLine().replaceAll("\\s+$", "").split(" ");
                int prereq = Integer.parseInt(secondMultipleInput[0]);
                int course = Integer.parseInt(secondMultipleInput[1]);

                result.addPrereq(prereq, course);
            } catch (IOException ex) {
                throw new RuntimeException(ex);
            }
        });
        if (result.isPossible())
            System.out.println("possible");
        else
            System.out.println("IMPOSSIBLE");

        bufferedReader.close();
    }
}