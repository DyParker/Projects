import Header from '../Header/Header.jsx';
import './Home.css';

export default function Home() {
    return (
        <>
        
        <Header num={1} />
            <section id="section0" className="container-fluid">
                <div className="container">
                    <div className="row">
                        <div className="col align-self-center">
                            <h2>WELCOME TO PROJECT MEALSWIPES </h2>
                            <p>We are the solution to the 119 billion pounds of annual food waste in the US.</p>
                        </div>
                    </div>
                </div>
            </section>
        </>
    )
}