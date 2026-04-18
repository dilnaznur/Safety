import { cp, mkdir, readdir } from 'node:fs/promises';
import path from 'node:path';

const sourceDir = path.resolve('static');
const outputDir = path.resolve('public');
const outputStaticDir = path.join(outputDir, 'static');

await mkdir(outputDir, { recursive: true });
await mkdir(outputStaticDir, { recursive: true });

// Keep app entry at root for Vercel static hosting.
await cp(path.join(sourceDir, 'index.html'), path.join(outputDir, 'index.html'));

// Keep assets under /static/* so existing absolute paths continue to work.
for (const item of await readdir(sourceDir, { withFileTypes: true })) {
	const src = path.join(sourceDir, item.name);
	const dest = path.join(outputStaticDir, item.name);
	if (item.name === 'index.html') {
		continue;
	}
	await cp(src, dest, { recursive: item.isDirectory() });
}

console.log(`Copied ${sourceDir}/index.html -> ${outputDir}/index.html`);
console.log(`Copied ${sourceDir} -> ${outputStaticDir}`);