import '../Header/Header.css';
import { Link } from "react-router-dom";
import { auth } from '../../firebase';
import { signOut } from 'firebase/auth';
import { useNavigate } from 'react-router-dom';


export default function Donor_Header() {
    const userSignOut = () => {
        signOut(auth).then(() => {
            console.log('sign out successful');
            handleSubmit();
        }).catch(error => console.log(error));
    }

    // This function "handleSubmit" allows us to submit data and link to a new page
    const navigate = useNavigate();

    const handleSubmit = event => {

        // Redirect to volunteer ui
        navigate('/');

    };
    return (

        <nav className="navbar navbar-expand-lg fixed-top navbar-light bg-light">
            <div className="container-fluid">
                <div className="navbar-brand"><Link to='/a_home'>MealSwipes</Link></div>
                <button className="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarSupportedContent" aria-controls="navbarSupportedContent" aria-expanded="false" aria-label="Toggle navigation"> {/* Hamburguer Menu */}
                    <span className="navbar-toggler-icon" />
                </button>
                <div className="collapse navbar-collapse" id="navbarSupportedContent">
                    <ul className="navbar-nav me-auto mb-2 mb-lg-0">
                        <li className="nav-item">
                            <Link to='/donorui'>Donor Homepage</Link>
                        </li>
                        <li className="nav-item">
                            <Link to='/a_about'>About</Link>
                        </li>
                        <li className="nav-item">
                            <Link to='/donorinfo'>Profile</Link>
                        </li>
                        <li className="nav-item">
                            <Link to='/' onClick={userSignOut}>Sign Out</Link>
                        </li>
                    </ul>
                </div>
            </div>
        </nav>
    )
}
