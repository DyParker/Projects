import Donor_Header from '../Donor_Header/Donor_Header.jsx';
import './Donor_Reviews.css';

export default function Rest_Reviews() {

    return (


        <body>
            <Donor_Header></Donor_Header>
            <section className="push"></section>


            <div class="row backarrow"><a href="volunteer.html"><i class="bi bi-arrow-left"></i></a></div>
            <div class="row">
                <div class="col card mt-3">
                    <div class="card-body">
                        <h5 class="card-title">Leave a review about the volunteers:</h5>
                        <a class="btn btn-success btn-green" href="#reviews-anchor" id="open-review-box">Write Review</a>
                        <div class="row" id="post-review-box">
                            <div class="col-md-12">
                                <form accept-charset="UTF-8" action="" method="post">
                                    <input id="ratings-hidden" name="rating" type="hidden" />
                                    <textarea class="form-control animated" cols="50" id="new-review" name="comment" placeholder="Enter your review here..." rows="5"></textarea>

                                    <div class="text-right">
                                        <div class="stars starrr" data-rating="0"></div>
                                        <a class="btn btn-danger btn-sm" href="#" id="close-review-box">
                                            <span class="glyphicon glyphicon-remove"></span>Cancel</a>
                                        <button class="btn btn-success btn-lg" type="submit">Save</button>
                                    </div>
                                </form>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="row">
                <div class="col card mt-3">
                    <div class="card-body">

                        <h5 class="card-title">Your evaluation</h5>
                        <div class="col-xs-12 col-md-6">
                            <div class="well well-sm">
                                <div class="row">
                                    <div class="col-xs-12 col-md-6 text-center">
                                        <h1 class="rating-num">
                                            4.0</h1>
                                        <div class="container">
                                            <span id="rateMe4" class="feedback"></span>
                                        </div>

                                        <script src="js/addons/rating.js"></script>
                                        <div>
                                            <span class="bi bi-star-fill"></span>
                                            <span class="bi bi-star-fill"></span>
                                            <span class="bi bi-star-fill"></span>
                                            <span class="bi bi-star-fill"></span>
                                            <span class="bi bi-star"></span>
                                        </div>
                                        <div>1,050,008 total</div>
                                    </div>
                                    <div class="col-xs-12 col-md-6">
                                        <div class="row rating-desc">
                                            <div class="col-xs-3 col-md-3 text-right">
                                                <span class="bi bi-star"></span>5
                                            </div>
                                            <div class="col-xs-8 col-md-9">
                                                <div class="progress progress-striped">
                                                    <div class="progress-bar progress-bar-success" role="progressbar" aria-valuenow="20"
                                                        aria-valuemin="0" aria-valuemax="100" >
                                                        <span class="sr-only">80%</span>
                                                    </div>
                                                </div>
                                            </div>

                                            <div class="col-xs-3 col-md-3 text-right">
                                                <span class="bi bi-star"></span>4
                                            </div>
                                            <div class="col-xs-8 col-md-9">
                                                <div class="progress">
                                                    <div class="progress-bar progress-bar-success" role="progressbar" aria-valuenow="20"
                                                        aria-valuemin="0" aria-valuemax="100" >
                                                        <span class="sr-only">60%</span>
                                                    </div>
                                                </div>
                                            </div>

                                            <div class="col-xs-3 col-md-3 text-right">
                                                <span class="bi bi-star"></span>3
                                            </div>
                                            <div class="col-xs-8 col-md-9">
                                                <div class="progress">
                                                    <div class="progress-bar progress-bar-info" role="progressbar" aria-valuenow="20"
                                                        aria-valuemin="0" aria-valuemax="100" >
                                                        <span class="sr-only">40%</span>
                                                    </div>
                                                </div>
                                            </div>

                                            <div class="col-xs-3 col-md-3 text-right">
                                                <span class="bi bi-star"></span>2
                                            </div>
                                            <div class="col-xs-8 col-md-9">
                                                <div class="progress">
                                                    <div class="progress-bar progress-bar-warning" role="progressbar" aria-valuenow="20"
                                                        aria-valuemin="0" aria-valuemax="100">
                                                        <span class="sr-only">20%</span>
                                                    </div>
                                                </div>
                                            </div>

                                            <div class="col-xs-3 col-md-3 text-right">
                                                <span class="bi bi-star"></span>1
                                            </div>
                                            <div class="col-xs-8 col-md-9">
                                                <div class="progress">
                                                    <div class="progress-bar progress-bar-danger" role="progressbar" aria-valuenow="80"
                                                        aria-valuemin="0" aria-valuemax="100">
                                                        <span class="sr-only">15%</span>
                                                    </div>
                                                </div>
                                            </div>

                                        </div>

                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>


                <div class="row">
                    <div class="col card mt-3">
                        <div class="card-body">
                            <h5 class="card-title">Text Reviews</h5>
                        </div>
                        <div class="carousel-reviews broun-block">
                            <div class="container">
                                <div class="row">
                                    <div id="carousel-reviews" class="carousel slide" data-ride="carousel">

                                        <div class="carousel-inner">
                                            <div class="item active">
                                                <div class="col-md-4 col-sm-6">
                                                    <div class="block-text rel zmin">
                                                        <a title="" href="#">Joe</a>
                                                        <div class="mark">Rating:
                                                            <span class="rating-input">
                                                                <span data-value="0" class="bi bi-star-fill"></span>
                                                                <span data-value="1" class="bi bi-star-fill"></span>
                                                                <span data-value="2" class="bi bi-star-fill"></span>
                                                                <span data-value="3" class="bi bi-star-fill"></span>
                                                                <span data-value="4" class="bi bi-star-fill"></span>
                                                                <span data-value="5" class="bi bi-star-fill"></span>
                                                            </span>
                                                        </div>
                                                        <p>Never before has there been a good film portrayal of ancient Greece's favourite myth. So why would Hollywood start now? This latest attempt at bringing the son of Zeus to the big screen is brought to us by X-Men: The last Stand director Brett Ratner. If the name of the director wasn't enough to dissuade ...</p>
                                                        <ins class="ab zmin sprite sprite-i-triangle block"></ins>
                                                    </div>

                                                </div>
                                                <div class="col-md-4 col-sm-6 hidden-xs">
                                                    <div class="block-text rel zmin">
                                                        <a title="" href="#">Chelsea</a>
                                                        <div class="mark">Rating:
                                                            <span class="rating-input">
                                                                <span data-value="0" class="bi bi-star-fill"></span>
                                                                <span data-value="1" class="bi bi-star-fill"></span>
                                                                <span data-value="2" class="bi bi-star-fill"></span>
                                                                <span data-value="3" class="bi bi-star-half"></span>
                                                                <span data-value="4" class="bi bi-star"></span>
                                                                <span data-value="5" class="bi bi-star"></span>
                                                            </span>
                                                        </div>
                                                        <p>The 2013 movie "The Purge" left a bad taste in all of our mouths as nothing more than a pseudo-slasher with a hamfisted plot, poor pacing, and a desperate attempt at "horror." Upon seeing the first trailer for "The Purge: Anarchy," my first and most immediate thought was "we really don't need another one of these."  </p>
                                                        <ins class="ab zmin sprite sprite-i-triangle block"></ins>
                                                    </div>

                                                </div>
                                                <div class="col-md-4 col-sm-6 hidden-sm hidden-xs">
                                                    <div class="block-text rel zmin">
                                                        <a title="" href="#">Ana</a>
                                                        <div class="mark">Rating:
                                                            <span class="rating-input">
                                                                <span data-value="0" class="bi bi-star-fill"></span>
                                                                <span data-value="1" class="bi bi-star-fill"></span>
                                                                <span data-value="2" class="bi bi-star-fill"></span>
                                                                <span data-value="3" class="bi bi-star-fill"></span>
                                                                <span data-value="4" class="bi bi-star"></span>
                                                                <span data-value="5" class="bi bi-star"></span>
                                                            </span>
                                                        </div>
                                                        <p>What a funny and entertaining film! I did not know what to expect, this is the fourth film in this vehicle's universe with the two Cars movies and then the first Planes movie. I was wondering if maybe Disney pushed it a little bit. However, Planes: Fire and Rescue is an entertaining film that is a fantastic sequel in this magical franchise. </p>
                                                        <ins class="ab zmin sprite sprite-i-triangle block"></ins>
                                                    </div>

                                                </div>
                                            </div>
                                        </div>
                                        <a class="left carousel-control" href="#carousel-reviews" role="button" data-slide="prev">
                                            <span class="glyphicon glyphicon-chevron-left"></span>
                                        </a>
                                        <a class="right carousel-control" href="#carousel-reviews" role="button" data-slide="next">
                                            <span class="glyphicon glyphicon-chevron-right"></span>
                                        </a>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </body>
    )
}