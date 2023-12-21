import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import {
  faComment,
  faPlusCircle,
  faUnderline,
} from "@fortawesome/free-solid-svg-icons";
import { db } from "../../firebase";
import { useCollectionData } from "react-firebase-hooks/firestore";
import Button from "react-bootstrap/Button";
import Modal from "react-bootstrap/Modal";
import Donor_UI from "./Donor_UI";

import firebase from "firebase/compat/app";
import "firebase/compat/firestore";
import "firebase/compat/auth";
import "firebase/compat/analytics";
import { useEffect, useRef, useState } from "react";
import {
  collection,
  query,
  where,
  orderBy,
  limit,
  getDocs,
  updateDoc,
  doc,
  setDoc,
} from "firebase/firestore";
import { TypeUnderline } from "react-bootstrap-icons";
import { useNavigate } from "react-router-dom";
import { deleteDoc } from "firebase/firestore";

export default function Delete_Task(props) {
  const firestore = firebase.firestore();

  const [showModal, setShowModal] = useState(false);

  const handleRejectClick = () => {
    setShowModal(false);
  };

  const modalStyle = {
    display: showModal ? "left" : "none",
    justifyContent: "left",
    position: "fixed",
    top: 100,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: "rgba(0, 0, 0, 0.5)",
    zIndex: 999,
  };

  const navigate = useNavigate();

  async function deleteTasks() {
    const docRef = doc(db, "tasks", props.path);
    await deleteDoc(docRef); // how do I actually reference the task?
    setShowModal(false);
    window.location.reload();
  }

  return (
    <>
      <div className="col">
        <button
          style={{
            whiteSpace: "nowrap",
            color: "maroon",
            textDecoration: "underline",
          }}
          onClick={() => setShowModal(true)}
        >
          Delete Task
        </button>
      </div>

      <Modal
        show={showModal}
        onHide={handleRejectClick}
        backdrop="static"
        keyboard={false}
      >
        <Modal.Header closeButton>
          <Modal.Title>Confirm Deletion?</Modal.Title>
        </Modal.Header>
        <Modal.Footer>
          <Button
            variant="secondary"
            onClick={deleteTasks}
            style={{ backgroundColor: "maroon" }}
          >
            Delete
          </Button>
          <Button variant="secondary" onClick={handleRejectClick}>
            Cancel
          </Button>
        </Modal.Footer>
      </Modal>
    </>
  );
}
