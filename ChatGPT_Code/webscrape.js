const puppeteer = require('puppeteer');
const fs = require('fs');

// Function to check if a URL is valid
function isValidUrl(url) {
  try {
    const urlObj = new URL(url);
    return urlObj.protocol.startsWith('http');
  } catch (error) {
    return false;
  }
}

// Function to perform BFS of links up to a specified depth
async function bfsLinks(initialUrl, depth) {
  const visited = new Set();
  const queue = [{ url: initialUrl, depth: 0 }];

  const foundLinks = [];

  while (queue.length > 0) {
    const { url, depth } = queue.shift();

    if (depth > 2) {
      continue; // Stop BFS at the specified depth
    }

    if (isValidUrl(url)) {
      visited.add(url);
      const browser = await puppeteer.launch({headless:'new'});
      const page = await browser.newPage();

      // Use Promise.race to set a timeout of 30 seconds for page.goto
      const pageLoadPromise = page.goto(url);
      const timeoutPromise = new Promise((_, reject) => setTimeout(() => reject(new Error('Timeout')), 20000));

      try {
        await Promise.race([pageLoadPromise, timeoutPromise]);

        const pageLinks = await page.$$eval('a', (links) => links.map((link) => link.href));
        foundLinks.push({ url, links: pageLinks });

        if (depth < 2) {
          for (const link of pageLinks) {
            queue.push({ url: link, depth: depth + 1 });
          }
        }
      } catch (error) {
        console.error(`Error on ${url}: ${error.message}`);
      } finally {
        await browser.close();
      }
    }
  }

  return foundLinks;
}

// URL of the initial webpage
const initialUrl = 'https://ombuds.emory.edu/'; // Replace with your desired URL

bfsLinks(initialUrl, 0)
  .then((foundLinks) => {
    fs.writeFileSync('foundLinks.json', JSON.stringify(foundLinks, null, 2));
    console.log('Links found and saved to foundLinks.json');
  })
  .catch((error) => {
    console.error('An error occurred:', error);
  });
