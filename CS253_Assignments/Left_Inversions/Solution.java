//THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING

//A TUTOR OR CODE WRITTEN BY OTHER STUDENTS - Dylan Parker

import java.io.*;
import java.math.*;
import java.security.*;
import java.text.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.function.*;
import java.util.regex.*;
import java.util.stream.*;
import static java.util.stream.Collectors.joining;
import static java.util.stream.Collectors.toList;

public class Solution {

    static int[] leftInv;

    static class Pair{

        int v; //a value from its array A
        int p; //its position, so A[p]==v
        Pair(int v, int p) {this.v=v; this.p=p;} //basic constructor
    }


    //basic merge that includes left inversion
    private static void merge(Pair[] P, Pair[] aux, int lo, int mid, int hi){

        int i=lo; int j=mid+1;
        for(int k=lo; k<=hi; k++){
            if(i>mid) aux[k]=P[j++];
            else if(j>hi) aux[k]=P[i++];
            else if((P[j].v < P[i].v))
            {
                aux[k]=P[j++];
                leftInv[aux[k].p] += (mid - i + 1); //updates left inversions
            }
            else aux[k]=P[i++];
        }
    }
    //basic mergesorting method
    private static void sort(Pair[] P, Pair[] aux, int lo, int hi){

        if(hi <= lo) return;
        int mid = lo+(hi-lo)/2;
        sort(aux, P, lo, mid);
        sort(aux, P, mid+1, hi);
        merge(P, aux, lo, mid, hi);
        return;
    }




    public static int[] computeL(int[] A) {

        int N=A.length;
        leftInv=new int[N];
        //making p and auxillary array
        Pair[] P = new Pair[N];
        Pair[] aux = new Pair[N];
        for (int i=0; i<N; ++i){
            P[i] = new Pair(A[i], i);
            aux[i] = new Pair(A[i], i);
        }

        sort(P, aux, 0, N-1);
        return leftInv;
    }


    public static void main(String[] args) throws IOException {
        // Read input array A. We avoid java.util.Scanner, for speed.
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        int N = Integer.parseInt(br.readLine()); // first line
        int[] A = new int[N];
        StringTokenizer st = new StringTokenizer(br.readLine()); // second line
        for (int i=0; i<N; ++i)
            A[i] = Integer.parseInt(st.nextToken());

        // Solve the problem!
        int[] L = computeL(A);

        // Print the output array L, again buffered for speed.
        PrintWriter out = new PrintWriter(System.out);
        out.print(L[0]);
        for (int i=1; i<N; ++i)
            // System.out.print here would be too slow!
            out.print(" " + L[i]);
        out.close();
    }
}
