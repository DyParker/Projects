//THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING
//A TUTOR OR CODE WRITTEN BY OTHER STUDENTS - Dylan Parker


import java.io.*;
import java.util.*;
import java.text.*;
import java.math.*;
import java.util.regex.*;

public class Solution {
    static class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;
        TreeNode() {}
        TreeNode(int val) { this.val = val; }
        TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }


    public static TreeNode deleteTree(TreeNode root) {
        //base case
        if(root==null){
            return null;
        }
        //calls itself
        root.left=deleteTree(root.left);
        root.right=(deleteTree(root.right));
        //sets return value and makes sure deletion works properly
        if(isEven(root.val) && root.left==null && root.right==null){
            return null;
        }
        //if it is not even then root is returned to itself
        return root;
    }

    //quick helper function
    public static boolean isEven(int a){
        return a%2==0;
    }

    public static void main(String[] args) throws IOException {
        // Read input array A. We avoid java.util.Scanner, for speed.
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        int N = Integer.parseInt(br.readLine()); // first line
        Integer[] A = new Integer[N];
        StringTokenizer st = new StringTokenizer(br.readLine()); // second line
        for (int i=0; i<N; ++i) {
            String s = st.nextToken();
            A[i] = (s.equals("null") ? null : Integer.parseInt(s));
        }

        // Create the input binary tree
        TreeNode root = new TreeNode();
        if (A[0] == null) {
            root = null;
        }
        else {
            int count = 0;
            Queue<TreeNode> q = new LinkedList<TreeNode>();
            root = new TreeNode(A[0]);
            q.add(root);
            TreeNode cur = null;
            for(int i = 1; i < A.length; i++){
                TreeNode node = new TreeNode();
                if (A[i] == null) {
                    node = null;
                } else {
                    node = new TreeNode(A[i]);
                }
                if(count == 0){
                    cur = q.poll();
                }
                if(count==0){
                    count++;
                    cur.left = node;
                }else {
                    count = 0;
                    cur.right = node;
                }
                if(A[i] != null){
                    q.add(node);
                }
            }
        }

        // Solve the problem!
        root = deleteTree(root);

        // Print the output binary tree, again buffered for speed.
        PrintWriter out = new PrintWriter(System.out);

        Queue<TreeNode> curr=new LinkedList<TreeNode>();
        Queue<TreeNode> next=new LinkedList<TreeNode>();

        if (root == null) out.print("null ");
        else {
            curr.add(root);
            next.add(root.left);
            next.add(root.right);
            out.print(root.val + " ");
            boolean end = false;
            while (!next.isEmpty()) {
                curr = next;
                next = new LinkedList<TreeNode>();
                while (!curr.isEmpty()) {
                    TreeNode temp = curr.poll();
                    if (temp == null) {
                        end = true;
                        for (TreeNode t : curr) {
                            if (t != null) {
                                end = false;
                                break;
                            }
                        }
                        if (end == true) {
                            for (TreeNode t : next) {
                                if (t != null) {
                                    end = false;
                                    break;
                                }
                            }
                        }
                        if (end == true) break;
                        out.print("null ");
                    } else {
                        out.print(temp.val + " ");
                        next.add(temp.left);
                        next.add(temp.right);
                    }
                }
                if (end == true) break;
            }
        }
        out.close();
    }
}